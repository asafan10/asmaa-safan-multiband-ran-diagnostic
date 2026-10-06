"""
STATUS: UNUSED / ABANDONED. Not referenced by run_experiments.py,
ablation_experiment.py, or any other script in this package, and no
result in the current paper (any table, figure, or equation) comes from
this file. It was an exploratory SOTA-architecture comparison against
related-work citations [14]/[15] that was dropped before the "Section
IV.G addendum" it mentions below was ever written into the paper. Kept
here for provenance only -- do not use it to check any claim about "the
real architecture" in the current paper; that is dqn_traffic_steering.py
(_QNetwork + DQNAgent, PyTorch path), used for Table VI, Table VII,
Section V.G's closed-loop substitution/K=150 retrain, and Appendix A.F.

NumPy Dueling Double-DQN agent -- the "one DRL-based HetNet band selector"
SOTA-architecture comparison (paper Section IV.G addendum / new subsection),
scoped as an in-testbed architecture ablation, NOT a reproduction of any
cited paper's reported numbers (see the three pre-commitments recorded in
conversation before this file was written).

Motivation / what is being tested: Table I cites [14] Liang et al. (2026),
a Dueling DDQN for HetNet RRM, and [15] He et al. (2017), a Dueling DQN for
Sub-6+mmWave band selection -- both DRL-for-RRM works whose headline
architectural choice, relative to a vanilla single-stream DQN, is (a) a
dueling value/advantage decomposition and (b) double-DQN's decoupling of
action SELECTION (by the online network) from action EVALUATION (by the
target network), which reduces the overestimation bias vanilla Q-learning
is prone to. This module reimplements that architectural combination as a
pure-NumPy variant of this project's own existing DQN pipeline
(dqn_traffic_steering.py's _MLPQApprox / DQNAgent), so it can be trained
and evaluated inside THIS paper's own environment, reward, and training
procedure -- not the cited papers' (different topology, physics model, and
reward, which are not reproducible without their code/data).

Pre-commitment #1 (hyperparameters fixed before training, not retuned
after seeing results): encoder hidden width is set to the SAME value as
the existing single-stream _MLPQApprox (min(hidden, 64)) specifically so
this is an architecture-only comparison, not an architecture+capacity
comparison. The one new hyperparameter double-DQN/dueling introduces --
the target hard-copy interval -- reuses DQNAgent's existing
target_hard_copy_every default (40 steps) rather than a separately-tuned
value.
"""
import numpy as np


class _DuelingMLPQApprox:
    """
    Pure-NumPy dueling Q-network: a shared single-hidden-layer encoder
    (identical capacity to _MLPQApprox's hidden layer) feeding two linear
    heads -- a scalar value stream V(s) and an act_dim-wide advantage
    stream A(s,.) -- combined as Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)),
    the standard dueling-DQN combination (Wang et al., 2016) used
    identically by [14]/[15]'s architectural description.

    Manual forward/backward (no autograd), single-sample SGD, matching the
    same predict/update_sgd/copy_weights_from interface _MLPQApprox and
    _LinearQApprox already expose in dqn_traffic_steering.py, so this
    class is a drop-in Q-function approximator for the same style of
    training loop -- only the double-DQN TARGET COMPUTATION (which
    requires seeing both the online and target network) lives outside
    this class, in DuelingDoubleDQNAgent.learn() below.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden
        # Same He-like initialization scheme as _MLPQApprox, using the
        # GLOBAL np.random state so np.random.seed(...) at the call site
        # controls initialization reproducibly (matching the fix already
        # applied to _MLPQApprox for the same reason).
        self.W_enc = np.random.standard_normal((hidden, obs_dim)) * np.sqrt(2.0 / obs_dim)
        self.b_enc = np.zeros(hidden)
        self.W_v = np.random.standard_normal((1, hidden)) * np.sqrt(2.0 / hidden)
        self.b_v = np.zeros(1)
        self.W_a = np.random.standard_normal((act_dim, hidden)) * np.sqrt(2.0 / hidden)
        self.b_a = np.zeros(act_dim)

    def _forward(self, state: np.ndarray):
        z1 = self.W_enc @ state + self.b_enc
        h = np.maximum(z1, 0.0)  # ReLU
        v = self.W_v @ h + self.b_v          # shape (1,)
        a = self.W_a @ h + self.b_a          # shape (act_dim,)
        q = v[0] + (a - a.mean())
        return q, z1, h, v, a

    def predict(self, state: np.ndarray) -> np.ndarray:
        q, _, _, _, _ = self._forward(state)
        return q

    def update_sgd(self, state, action, target, lr=7e-4):
        """Single-sample SGD update toward `target` for Q(state, action),
        matching _MLPQApprox.update_sgd's interface and semantics -- the
        TD error is defined against whatever target the caller computed
        (vanilla or double-DQN); this method does not know or care which."""
        q, z1, h, v, a = self._forward(state)
        err = target - q[action]

        n = self.act_dim
        # dQ[action]/dv = 1 ; dQ[action]/da[action] = 1 - 1/n ; dQ[action]/da[j!=action] = -1/n
        dL_dv = -err                      # dL/dQ[action] = -err, times dQ/dv = 1
        dL_da = np.full(n, err / n)
        dL_da[action] += -err

        dW_v = dL_dv * h.reshape(1, -1)
        db_v = np.array([dL_dv])
        dW_a = np.outer(dL_da, h)
        db_a = dL_da.copy()

        dh = dL_dv * self.W_v[0] + self.W_a.T @ dL_da
        dz1 = dh * (z1 > 0)               # ReLU derivative
        dW_enc = np.outer(dz1, state)
        db_enc = dz1

        self.W_v -= lr * dW_v; self.b_v -= lr * db_v
        self.W_a -= lr * dW_a; self.b_a -= lr * db_a
        self.W_enc -= lr * dW_enc; self.b_enc -= lr * db_enc
        return err ** 2

    def copy_weights_from(self, other: "_DuelingMLPQApprox"):
        self.W_enc, self.b_enc = other.W_enc.copy(), other.b_enc.copy()
        self.W_v, self.b_v = other.W_v.copy(), other.b_v.copy()
        self.W_a, self.b_a = other.W_a.copy(), other.b_a.copy()


class DuelingDoubleDQNAgent:
    """
    Drop-in replacement for dqn_traffic_steering.DQNAgent (same
    select_action / _greedy_action / store / learn / eps interface, so
    dqn_traffic_steering.DQNTrainer can train it unmodified), backed by
    _DuelingMLPQApprox and using a DOUBLE-DQN target: the ONLINE network
    selects the next-state's best action, the TARGET network evaluates it
    -- decoupling selection from evaluation, the mechanism double-DQN uses
    to reduce the overestimation bias of vanilla Q-learning's
    max_a Q_target(s', a).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        hidden: int = 128,
        target_hard_copy_every: int = 40,  # same default as DQNAgent's numpy path
        **_kwargs,
    ):
        from dqn_traffic_steering import ReplayBuffer  # reuse, not reimplement

        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.lr = lr
        self.gamma = gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buffer_size)
        self.train_step = 0
        self.target_hard_copy_every = target_hard_copy_every
        self.numpy_backend = "dueling_double_mlp"
        self.backend = "numpy"

        h = min(hidden, 64)  # same capacity cap as _MLPQApprox -- see module docstring
        self.q_net = _DuelingMLPQApprox(obs_dim, act_dim, hidden=h)
        self.tgt_net = _DuelingMLPQApprox(obs_dim, act_dim, hidden=h)
        self.tgt_net.copy_weights_from(self.q_net)

    def select_action(self, state: np.ndarray) -> int:
        import random
        if random.random() < self.eps:
            return random.randrange(self.act_dim)
        return self._greedy_action(state)

    def _greedy_action(self, state: np.ndarray) -> int:
        return int(np.argmax(self.q_net.predict(state)))

    def store(self, state, action, reward, next_state, done):
        self.replay.push(
            np.array(state, dtype=np.float32), action, reward,
            np.array(next_state, dtype=np.float32), done,
        )

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        total = 0.0
        for t in batch:
            # Double-DQN target: online net picks the action, target net
            # evaluates it -- the one substantive algorithmic difference
            # from the existing single-stream agent's vanilla
            # max_a Q_target(s', a) target in _numpy_learn.
            next_q_online = self.q_net.predict(t.next_state)
            best_next_action = int(np.argmax(next_q_online))
            next_q_target = self.tgt_net.predict(t.next_state)[best_next_action]
            target = t.reward + self.gamma * next_q_target * (1 - t.done)
            total += self.q_net.update_sgd(t.state, t.action, target, lr=self.lr)

        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.train_step += 1
        if self.train_step % self.target_hard_copy_every == 0:
            self.tgt_net.copy_weights_from(self.q_net)
        return total / len(batch)
