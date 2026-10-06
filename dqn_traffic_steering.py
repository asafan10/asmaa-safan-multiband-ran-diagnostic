"""
=============================================================================
  MODULE 4 — Deep Q-Network (DQN) for Traffic Steering
=============================================================================
Features:
  • Custom multi-band RAN gym-style environment
  • Dueling DQN with Double-DQN targets (PyTorch)
  • NumPy linear-approximation fallback when PyTorch is absent
  • DQNAgent is ALWAYS defined regardless of backend
  • Experience replay buffer, epsilon-greedy exploration
  • Training loop with ASCII reward curve and band-selection stats
=============================================================================

  >>> Fixed bug: this module used to embed its OWN copy of Band,
  PropagationModel, and HierarchicalMultiBandPlanner instead of
  importing them from multiband_planning.py. That copy had drifted:
  it still used THz range = 30 m / mmWave range = 200 m / THz max
  velocity = 1.5 m/s / mmWave max velocity = 10.0 m/s, all of which
  had since been updated to 80 m / 300 m / 5.0 m/s / 20.0 m/s in the
  canonical multiband_planning.py. That meant the DQN environment was
  training and evaluating against a DIFFERENT band-eligibility model
  than Modules 1-3 actually use — silently invalidating the paper's
  "closed-loop, four modules share one system model" framing.

  This version imports everything shared from multiband_planning.py.
  There is now exactly one place these thresholds live, so this
  specific bug class cannot recur. <<<

  >>> Reward-weight tuning history <<<
  EnvConfig's reward weights below were originally hand-tuned through a
  sequence of inline "FIX:" comments (e.g. "w_latency 0.25->0.10, was
  too dominant, caused mmWave collapse"). That tuning history is
  preserved in REWARD_WEIGHT_TUNING_LOG below for traceability, and the
  weights are now defined as a named RewardWeights dataclass that
  run_experiments.py's `sweep_reward_weights()` can grid-search over —
  turning "we hand-tuned this" into a reproducible ablation.
=============================================================================
"""

import math
import random
import numpy as np
from dataclasses import dataclass, field
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from multiband_planning import (
    Band, BaseStation, UserEquipment, Position, PropagationModel,
    HierarchicalMultiBandPlanner, LATENCY_MS, HO_PENALTY_MS, MAX_HO_PENALTY_MS,
    LOAD_PENALTY_COEFF,
)

# ── PyTorch — optional ────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[DQN] PyTorch not found — using NumPy fallback (single-hidden-layer MLP by default; "
          "pass numpy_backend='linear' to DQNAgent for the older linear-only approximator).")


# =============================================================================
# MODULE 4 — DQN Core
# =============================================================================

BAND_LIST = list(Band)
N_BANDS   = len(BAND_LIST)
BAND_IDX  = {b: i for i, b in enumerate(BAND_LIST)}

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


# ─────────────────────────────────────────────────────────────────────────────
# 4.1  Reward weights — documented tuning history + environment config
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable log of the manual tuning steps that produced the current
# defaults below, preserved for traceability (this used to live as
# scattered inline "FIX:" comments):
REWARD_WEIGHT_TUNING_LOG = """
  w_tput     : 0.40 -> 0.45
  w_latency  : 0.25 -> 0.10   (was too dominant; caused mmWave collapse)
  w_load_bal : 0.20 -> 0.30   (encourage genuine load distribution)
  w_ho_cost  : 0.15 -> 0.10
  w_entropy  : 0.15 -> 0.20   (stronger 3-band exploration after 5
                               consecutive THz misses were observed)
  These were NOT independently ablated — see run_experiments.py's
  sweep_reward_weights() to grid-search each weight and confirm (or
  revise) these choices with actual data before citing them as final.
"""


@dataclass
class RewardWeights:
    w_tput:     float = 0.45
    w_latency:  float = 0.10
    w_load_bal: float = 0.30
    w_ho_cost:  float = 0.10
    w_entropy:  float = 0.20


@dataclass
class EnvConfig:
    n_ues:        int   = 20
    area_m:       float = 500.0
    sub6_count:   int   = 3
    mmwave_count: int   = 8
    thz_count:    int   = 15
    max_steps:    int   = 200
    tick_s:       float = 0.1
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


class MultiBandRANEnv:
    """
    >>> Fixed bug: stale load signal <<<
    A previous version of step() set `ue.assigned_band = new_band` directly
    without ever updating any BaseStation's `active_ues` list. Since
    `bs.load` is derived from `len(bs.active_ues)`, that meant the load
    values feeding BOTH the observation (`_observe_ue`'s `loads` features)
    AND the reward (`r_load`, the load-balance term) were frozen at
    whatever `_spawn_ues()`'s one-time `planner.run_planning()` produced —
    they never actually changed in response to the agent's actions. The
    agent could learn to "prefer" a band based on a load signal that had
    no real relationship to what it was doing.

    This version routes every action through `_apply_action()`, which
    actually connects/disconnects the UE to a real BaseStation and updates
    `active_ues`, so load — and therefore both the observation and the
    reward — reflects the real, current consequence of the policy's
    actions. This also answers a question the review raised directly:
    "what happens when the DQN selects a band with no reachable BS?" —
    see `_apply_action`'s return value and the reward penalty below.

    >>> Scope note on the review's non-stationarity critique <<<
    The review's deeper point — that formulating this as independent
    per-UE single-agent MDPs is not fully valid because UEs act
    simultaneously and affect each other's environment (true multi-agent
    non-stationarity) — is NOT solved by this fix, and isn't something a
    local patch can solve; the review's own proposed fix is a full CTDE
    (centralized-training, decentralized-execution) multi-agent redesign,
    which is a substantial separate undertaking, not a bug fix. What this
    fix DOES do is make the existing single-agent approximation internally
    consistent (actions have real, observable consequences within an
    episode) rather than leaving a load signal that was disconnected from
    the agent's own behavior — a correctness bug independent of whether
    the single-agent framing itself is the right modeling choice.
    """
    OBS_DIM_PER_UE = 9
    INFEASIBLE_ACTION_PENALTY = -1.0  # full-scale penalty; reward is clipped to [-1, 1]

    def __init__(self, cfg: EnvConfig = None):
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.prop    = PropagationModel()
        self.planner = HierarchicalMultiBandPlanner()
        self.planner.deploy_grid(
            area_m=self.cfg.area_m, sub6_count=self.cfg.sub6_count,
            mmwave_count=self.cfg.mmwave_count, thz_count=self.cfg.thz_count,
        )
        self.planner._bs_index = {bs.bs_id: bs for bs in self.planner.base_stations}
        self.ues:         List[UserEquipment] = []
        self.app_types:   Dict[int, int]      = {}
        self.step_count   = 0
        self.obs_dim      = self.OBS_DIM_PER_UE
        self.act_dim      = N_BANDS
        self._prev_bands: Dict[int, Optional[Band]] = {}
        self._ue_pointer  = 0

    def _spawn_ues(self):
        for i in range(self.cfg.n_ues):
            pos = Position(random.uniform(0, self.cfg.area_m),
                            random.uniform(0, self.cfg.area_m))
            ue = UserEquipment(ue_id=i, position=pos, velocity=random.uniform(0, 30))
            self.ues.append(ue)
            self.app_types[i] = random.randint(0, 4)
        self.planner.ues = self.ues
        self.planner.run_planning()

    def _move_ues(self):
        for ue in self.ues:
            angle = random.uniform(0, 2 * math.pi)
            ue.position.x = max(0, min(self.cfg.area_m,
                ue.position.x + ue.velocity * math.cos(angle) * self.cfg.tick_s))
            ue.position.y = max(0, min(self.cfg.area_m,
                ue.position.y + ue.velocity * math.sin(angle) * self.cfg.tick_s))

    def _observe_ue(self, ue: UserEquipment) -> np.ndarray:
        snrs, loads = [], []
        for band in BAND_LIST:
            band_bss = [b for b in self.planner.base_stations if b.band == band]
            if band_bss:
                best_snr = max(self.prop.snr_db(bs, ue) for bs in band_bss)
                avg_load = float(np.mean([bs.load for bs in band_bss]))
            else:
                best_snr, avg_load = -99.0, 0.0
            snrs.append(np.clip((best_snr + 20) / 60, 0, 1))
            loads.append(avg_load)
        vel_norm = np.clip(ue.velocity / 50.0, 0, 1)
        app_norm = self.app_types.get(ue.ue_id, 0) / 4.0
        return np.array(snrs + loads + [vel_norm, app_norm, 1.0], dtype=np.float32)

    def _disconnect(self, ue: UserEquipment):
        if ue.assigned_bs is not None and ue.assigned_bs in self.planner._bs_index:
            old_bs = self.planner._bs_index[ue.assigned_bs]
            if ue.ue_id in old_bs.active_ues:
                old_bs.active_ues.remove(ue.ue_id)
        ue.assigned_bs, ue.assigned_band, ue.throughput_mbps = None, None, 0.0

    def _apply_action(self, ue: UserEquipment, band: Band) -> Optional[BaseStation]:
        """
        Actually connect `ue` to the best real base station in `band`
        (by raw SNR among BSs of that band, no range/velocity gating —
        the agent is meant to LEARN feasibility from the reward penalty
        below, not have it hard-coded as in the hierarchical planner).
        Returns the connected BaseStation, or None if no BS of that band
        exists at all in the deployment (a genuinely infeasible action).
        """
        self._disconnect(ue)
        band_bss = [b for b in self.planner.base_stations if b.band == band]
        if not band_bss:
            return None
        best_bs = max(band_bss, key=lambda b: self.prop.snr_db(b, ue))
        best_bs.active_ues.append(ue.ue_id)
        ue.assigned_bs   = best_bs.bs_id
        ue.assigned_band = band
        n = max(len(best_bs.active_ues), 1)
        ue.throughput_mbps = self.prop.shannon_capacity_mbps(best_bs, ue) / n
        return best_bs

    def _compute_reward(self, ue: UserEquipment, new_band: Band, connected_bs: Optional[BaseStation]) -> float:
        w = self.cfg.reward_weights

        if connected_bs is None:
            # No base station of the selected band exists anywhere in the
            # deployment — a fully infeasible action. Answers the review's
            # explicit question about what happens in this case: it is
            # scored as a hard failure, not silently ignored.
            return self.INFEASIBLE_ACTION_PENALTY

        r_tput = np.clip(ue.throughput_mbps / 10000.0, 0, 1)

        lat   = LATENCY_MS[new_band]
        r_lat = 1.0 - np.clip(lat / 100.0, 0, 1)

        all_loads = [bs.load for bs in self.planner.base_stations]
        r_load    = 1.0 - float(np.std(all_loads)) if all_loads else 0.0

        # Handover cost: magnitude-based (band-pair-specific protocol/beam-
        # alignment overhead from HO_PENALTY_MS — shared with Module 2's
        # TOPSIS scorer), NOT a flat "0.3 if band changed" constant. A
        # Sub-6<->THz handover now correctly costs more than Sub-6<->mmWave.
        prev = self._prev_bands.get(ue.ue_id)
        if prev is None or prev == new_band:
            ho_cost = 0.0
        else:
            ho_cost = HO_PENALTY_MS.get((prev, new_band), MAX_HO_PENALTY_MS) / MAX_HO_PENALTY_MS

        # Entropy bonus — reward diversity in band selection across UEs, to
        # counteract the policy's tendency to collapse onto whichever band
        # gets a marginally higher throughput/latency reward on average.
        band_counts = {b: 0 for b in Band}
        for u in self.ues:
            if u.assigned_band:
                band_counts[u.assigned_band] += 1
        total_assigned = max(sum(band_counts.values()), 1)
        usage_frac      = band_counts[new_band] / total_assigned
        r_entropy       = 1.0 - usage_frac   # rarer bands score higher

        reward = (w.w_tput     * r_tput
                + w.w_latency  * r_lat
                + w.w_load_bal * r_load
                + w.w_entropy  * r_entropy
                - w.w_ho_cost  * ho_cost)
        return float(np.clip(reward, -1, 1))

    def reset(self) -> Tuple[int, np.ndarray]:
        self.ues.clear()
        self.app_types.clear()
        self._prev_bands.clear()
        self.planner.ues = []
        for bs in self.planner.base_stations:
            bs.active_ues.clear()
        self.step_count  = 0
        self._ue_pointer = 0
        self._spawn_ues()
        ue = self.ues[self._ue_pointer]
        return ue.ue_id, self._observe_ue(ue)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        ue = self.ues[self._ue_pointer]
        new_band = BAND_LIST[action]
        self._prev_bands[ue.ue_id] = ue.assigned_band
        connected_bs = self._apply_action(ue, new_band)
        reward = self._compute_reward(ue, new_band, connected_bs)
        self._ue_pointer = (self._ue_pointer + 1) % len(self.ues)
        if self._ue_pointer == 0:
            self._move_ues()
            self.step_count += 1
        done = self.step_count >= self.cfg.max_steps
        next_ue  = self.ues[self._ue_pointer]
        next_obs = self._observe_ue(next_ue)
        return next_obs, reward, done, {"ue_id": ue.ue_id, "band": new_band.value, "step": self.step_count}


# ─────────────────────────────────────────────────────────────────────────────
# 4.2  Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# 4.3  Neural network (PyTorch) — defined only when torch is available
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:
    class _QNetwork(nn.Module):
        """Dueling DQN — value stream + advantage stream."""
        def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),  nn.ReLU(),
            )
            self.value_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
            self.adv_head   = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, act_dim))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            enc = self.encoder(x)
            val = self.value_head(enc)
            adv = self.adv_head(enc)
            return val + (adv - adv.mean(dim=-1, keepdim=True))


# ─────────────────────────────────────────────────────────────────────────────
# 4.4  NumPy fallbacks — always available (no PyTorch required)
# ─────────────────────────────────────────────────────────────────────────────

class _LinearQApprox:
    """Pure-NumPy linear Q-function: Q(s,a) = W_a . s + b_a"""
    def __init__(self, obs_dim: int, act_dim: int):
        self.W = np.random.randn(act_dim, obs_dim) * 0.01
        self.b = np.zeros(act_dim)

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def update_sgd(self, state, action, target, lr=7e-4):
        err = target - self.predict(state)[action]
        self.W[action] += lr * err * state
        self.b[action] += lr * err
        return err ** 2

    def copy_weights_from(self, other: "_LinearQApprox"):
        self.W = other.W.copy()
        self.b = other.b.copy()


class _MLPQApprox:
    """
    One-hidden-layer NumPy Q-network (Linear -> ReLU -> Linear), used as
    the default fallback when PyTorch isn't installed — noticeably more
    capacity than _LinearQApprox (which is a literal Q(s,a)=W.s+b with no
    non-linearity at all). Manual forward/backward since there's no
    autograd here; this is single-sample SGD, matching the interface
    _LinearQApprox already exposes (predict / update_sgd /
    copy_weights_from) so DQNAgent can use either interchangeably.

    Added specifically to separate two different explanations for a weak
    NumPy-backend DQN result: "the function approximator doesn't have
    enough capacity" (this class should measurably help) vs. "the
    observation/reward/action design itself doesn't give the agent enough
    signal to beat simple heuristics" (this class should NOT help much,
    which would be a more specific and more concerning finding).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        # NOTE: uses the GLOBAL np.random state (np.random.standard_normal),
        # not an independently-constructed np.random.default_rng() -- an
        # earlier version used default_rng() with no seed argument, which
        # draws from fresh OS entropy every time regardless of any prior
        # np.random.seed(seed) call. That silently broke reproducibility
        # for every experiment using this fallback (i.e. every DQN result
        # in this codebase, since PyTorch is unavailable in the evaluation
        # environment) -- identical seeds could still produce different
        # weight initializations, and therefore different results, on
        # every run. Matches _LinearQApprox's approach (np.random.randn),
        # which was already seed-respecting.
        self.W1 = np.random.standard_normal((hidden, obs_dim)) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.standard_normal((act_dim, hidden)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(act_dim)

    def _forward(self, state: np.ndarray):
        z1 = self.W1 @ state + self.b1
        h  = np.maximum(z1, 0.0)          # ReLU
        q  = self.W2 @ h + self.b2
        return q, z1, h

    def predict(self, state: np.ndarray) -> np.ndarray:
        q, _, _ = self._forward(state)
        return q

    def update_sgd(self, state, action, target, lr=7e-4):
        q, z1, h = self._forward(state)
        err = target - q[action]          # TD error for the taken action only

        # d(0.5*err^2)/dparam via manual backprop; only the chosen action's
        # output row gets a nonzero gradient at the output layer.
        dW2 = np.zeros_like(self.W2); db2 = np.zeros_like(self.b2)
        dW2[action] = -err * h
        db2[action] = -err
        dh  = -err * self.W2[action]
        dz1 = dh * (z1 > 0)               # ReLU derivative
        dW1 = np.outer(dz1, state)
        db1 = dz1

        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        return err ** 2

    def copy_weights_from(self, other: "_MLPQApprox"):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()


# ─────────────────────────────────────────────────────────────────────────────
# 4.5  DQNAgent — ALWAYS DEFINED (backend chosen at runtime)
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent.
    Uses Dueling DQN (PyTorch) when available, else NumPy linear fallback.
    DQNAgent is always importable regardless of whether PyTorch is installed.
    """

    def __init__(
        self,
        obs_dim:     int,
        act_dim:     int,
        lr:          float = 7e-4,
        gamma:       float = 0.99,
        eps_start:   float = 1.0,
        eps_end:     float = 0.05,
        eps_decay:   float = 0.995,
        batch_size:  int   = 64,
        tau:         float = 0.005,
        buffer_size: int   = 50_000,
        hidden:      int   = 128,
        target_update_every: int = 5,     # torch soft-update cadence (steps)
        target_hard_copy_every: int = 40, # numpy hard-copy cadence (steps)
        numpy_backend: str = "mlp",       # "mlp" (default) or "linear" — see _MLPQApprox docstring
        **_kwargs,                        # absorb unknown kwargs safely
    ):
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = gamma
        self.eps        = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.batch_size = batch_size
        self.tau        = tau
        self.replay     = ReplayBuffer(buffer_size)
        self.train_step = 0
        self.target_update_every    = target_update_every
        self.target_hard_copy_every = target_hard_copy_every
        self.numpy_backend          = numpy_backend

        if TORCH_AVAILABLE:
            self.backend = "torch"
            self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_net   = _QNetwork(obs_dim, act_dim, hidden).to(self.device)
            self.tgt_net = _QNetwork(obs_dim, act_dim, hidden).to(self.device)
            self.tgt_net.load_state_dict(self.q_net.state_dict())
            self.tgt_net.eval()
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        else:
            self.backend = "numpy"
            self.device  = None
            if numpy_backend == "linear":
                self.q_net   = _LinearQApprox(obs_dim, act_dim)
                self.tgt_net = _LinearQApprox(obs_dim, act_dim)
            else:
                self.q_net   = _MLPQApprox(obs_dim, act_dim, hidden=min(hidden, 64))
                self.tgt_net = _MLPQApprox(obs_dim, act_dim, hidden=min(hidden, 64))
            self.tgt_net.copy_weights_from(self.q_net)

    # ── Action selection ──────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.act_dim)
        return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray) -> int:
        if self.backend == "torch":
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return int(self.q_net(s).argmax(dim=1).item())
        return int(np.argmax(self.q_net.predict(state)))

    # ── Learning ──────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        self.replay.push(
            np.array(state,      dtype=np.float32), action, reward,
            np.array(next_state, dtype=np.float32), done,
        )

    def learn(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None
        batch       = self.replay.sample(self.batch_size)
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch], dtype=np.int64)
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)

        loss = (self._torch_learn(states, actions, rewards, next_states, dones)
                if self.backend == "torch"
                else self._numpy_learn(states, actions, rewards, next_states, dones))

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.train_step += 1
        if self.backend == "torch" and self.train_step % self.target_update_every == 0:
            self._soft_update()
        return loss

    def _torch_learn(self, states, actions, rewards, next_states, dones) -> float:
        S  = torch.FloatTensor(states).to(self.device)
        A  = torch.LongTensor(actions).to(self.device)
        R  = torch.FloatTensor(rewards).to(self.device)
        S_ = torch.FloatTensor(next_states).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)
        q_vals = self.q_net(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a  = self.q_net(S_).argmax(dim=1)
            next_q  = self.tgt_net(S_).gather(1, next_a.unsqueeze(1)).squeeze(1)
            targets = R + self.gamma * next_q * (1 - D)
        loss = F.huber_loss(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)
        self.optimizer.step()
        return float(loss.item())

    def _numpy_learn(self, states, actions, rewards, next_states, dones) -> float:
        total = 0.0
        for i in range(len(states)):
            q_next = np.max(self.tgt_net.predict(next_states[i]))
            target = rewards[i] + self.gamma * q_next * (1 - dones[i])
            total += self.q_net.update_sgd(states[i], actions[i], target)
        if self.train_step % self.target_hard_copy_every == 0:
            self.tgt_net.copy_weights_from(self.q_net)
        return total / len(states)

    def _soft_update(self):
        for p, pt in zip(self.q_net.parameters(), self.tgt_net.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)


# ─────────────────────────────────────────────────────────────────────────────
# 4.6  Training config & trainer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    n_episodes:    int  = 200
    max_steps:     int  = 200  # NOTE: no longer used by DQNTrainer (see _run_episode) --
                                # episode length is controlled entirely by EnvConfig.max_steps.
                                # Kept as a field for backward compatibility with any external
                                # code constructing TrainingConfig(max_steps=...); setting it no
                                # longer has any effect on training.
    eval_interval: int  = 20
    log_interval:  int  = 10
    save_best:     bool = True


class DQNTrainer:
    def __init__(
        self,
        env_cfg:      EnvConfig      = None,
        train_cfg:    TrainingConfig = None,
        agent_kwargs: Dict           = None,
    ):
        self.env_cfg   = env_cfg   if env_cfg   is not None else EnvConfig()
        self.train_cfg = train_cfg if train_cfg is not None else TrainingConfig()
        self.env       = MultiBandRANEnv(self.env_cfg)

        self.agent = DQNAgent(
            obs_dim=self.env.obs_dim, act_dim=self.env.act_dim,
            **(agent_kwargs or {}),
        )

        self.episode_rewards:  List[float] = []
        self.episode_losses:   List[float] = []
        self.eval_rewards:     List[float] = []
        self.best_eval_reward: float       = -float("inf")

    def _run_episode(self, train: bool = True) -> Tuple[float, float]:
        _, obs = self.env.reset()
        total_reward, total_loss, loss_count = 0.0, 0.0, 0
        # Loop bound derived from env_cfg.max_steps (the value that actually
        # controls MultiBandRANEnv.step()'s `done` flag), not train_cfg.max_steps
        # -- an earlier version used train_cfg.max_steps here, a SEPARATE field
        # that has to be kept manually in sync with env_cfg.max_steps. It never
        # caused wrong episode lengths in this codebase's own experiments only
        # because train_cfg.max_steps's default (200) always happened to exceed
        # every env_cfg.max_steps actually used (30/50) -- `done` fired first
        # and the loop broke early every time. But if env_cfg.max_steps were
        # ever set above train_cfg.max_steps, this loop would silently cut
        # episodes short before the environment's own completion condition
        # fired. Deriving the bound from env_cfg directly removes that
        # possibility rather than relying on the two fields staying in sync.
        max_iters = self.env_cfg.max_steps * self.env_cfg.n_ues
        for _ in range(max_iters):
            action = (self.agent.select_action(obs) if train
                      else self.agent._greedy_action(obs))
            next_obs, reward, done, _ = self.env.step(action)
            if train:
                self.agent.store(obs, action, reward, next_obs, done)
                loss = self.agent.learn()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
            total_reward += reward
            obs = next_obs
            if done:
                break
        return total_reward, total_loss / max(loss_count, 1)

    def train(self, verbose: bool = True):
        if verbose:
            backend = (f"PyTorch (Dueling DQN) on {self.agent.device}" if TORCH_AVAILABLE
                       else f"NumPy fallback ({self.agent.numpy_backend})")
            print("\n" + "=" * 65)
            print("  DEEP Q-NETWORK — TRAFFIC STEERING TRAINING")
            print(f"  Backend  : {backend}")
            print(f"  Episodes : {self.train_cfg.n_episodes}  "
                  f"UEs: {self.env_cfg.n_ues}  "
                  f"obs_dim: {self.env.obs_dim}  act_dim: {self.env.act_dim}")
            print("=" * 65)

        for ep in range(1, self.train_cfg.n_episodes + 1):
            ep_reward, ep_loss = self._run_episode(train=True)
            self.episode_rewards.append(ep_reward)
            self.episode_losses.append(ep_loss)

            if verbose and ep % self.train_cfg.log_interval == 0:
                w    = self.train_cfg.log_interval
                avgr = float(np.mean(self.episode_rewards[-w:]))
                avgl = float(np.mean(self.episode_losses[-w:]))
                print(f"  Ep {ep:4d}/{self.train_cfg.n_episodes}  |  "
                      f"AvgReward: {avgr:+.4f}  |  "
                      f"AvgLoss: {avgl:.6f}  |  "
                      f"eps: {self.agent.eps:.3f}")

            if ep % self.train_cfg.eval_interval == 0:
                eval_r, _ = self._run_episode(train=False)
                self.eval_rewards.append(eval_r)
                if eval_r > self.best_eval_reward:
                    self.best_eval_reward = eval_r
                if verbose:
                    tag = " <- best" if eval_r == self.best_eval_reward else ""
                    print(f"  *** EVAL ep {ep:4d}: reward = {eval_r:+.4f}{tag}")

        n = len(self.episode_rewards)
        g = (np.mean(self.episode_rewards[-10:]) -
             np.mean(self.episode_rewards[:10])) if n >= 10 else 0
        if verbose:
            print("\n" + "=" * 65)
            print(f"  Training complete.")
            print(f"  Best eval reward : {self.best_eval_reward:+.4f}")
            print(f"  Final eps        : {self.agent.eps:.4f}")
            print(f"  Reward gain      : {g:+.4f}")
            print("=" * 65)

    def plot_training_curves(self):
        rewards = self.episode_rewards
        n = len(rewards)
        if n == 0:
            return
        width, height = 60, 10
        r_min, r_max = min(rewards), max(rewards)
        r_range = r_max - r_min + 1e-9
        print("\n  Training Reward Curve:")
        print("  " + "-" * (width + 2))
        for row in range(height, -1, -1):
            threshold = r_min + (row / height) * r_range
            line = ""
            for col in range(width):
                idx = int(col / width * n)
                line += "#" if rewards[min(idx, n - 1)] >= threshold else " "
            label = f"{threshold:+.2f}" if row % 3 == 0 else "     "
            print(f"  {label}|{line}|")
        print(f"  {'':6}-{'-'*width}")
        print(f"  {'':6}Ep 1{' '*(width-8)}Ep {n}")

    def band_selection_stats(self) -> Dict[str, int]:
        _, obs = self.env.reset()
        counts = {b.value: 0 for b in Band}
        # Same fix as _run_episode(): derive the loop bound from env_cfg,
        # not the separate train_cfg.max_steps field.
        max_iters = self.env_cfg.max_steps * self.env_cfg.n_ues
        for _ in range(max_iters):
            action = self.agent._greedy_action(obs)
            obs, _, done, info = self.env.step(action)
            counts[info["band"]] += 1
            if done:
                break
        total = sum(counts.values()) or 1
        print("\n  Greedy Band Selection Distribution:")
        for band, cnt in counts.items():
            bar = "#" * int(30 * cnt / total)
            print(f"    {band:<15}: {bar} {cnt:5d} ({100*cnt/total:.1f}%)")
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # eps_decay derived from the training budget (same fix applied to
    # run_experiments.py's compare_dqn_vs_heuristics() and
    # sweep_reward_weights()) -- this demo's own hardcoded eps_decay=0.988
    # reached the exploration floor at episode ~0.5 of a planned 60-episode
    # run (10 UEs x 50 steps/episode = 500 steps/episode), meaning this
    # demo trained in near-zero-exploration mode for essentially its
    # entire duration. Fixed the same way: floor reached at ~80% of total
    # training steps instead.
    _n_ues, _max_steps, _n_episodes = 10, 50, 60
    _total_steps = _n_ues * _max_steps * _n_episodes
    _target_steps = max(int(0.8 * _total_steps), 1)
    _eps_decay = (0.05 / 1.0) ** (1.0 / _target_steps)

    trainer = DQNTrainer(
        env_cfg=EnvConfig(n_ues=_n_ues, max_steps=_max_steps),
        train_cfg=TrainingConfig(n_episodes=_n_episodes, log_interval=10, eval_interval=20),
        agent_kwargs=dict(lr=7e-4, gamma=0.99, eps_decay=_eps_decay, batch_size=32, hidden=64),
    )
    trainer.train()
    trainer.plot_training_curves()
    trainer.band_selection_stats()
