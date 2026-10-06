"""
Ablation row 4: A1+2+3 with Module 4 (DQN) substituted for Module 2 (TOPSIS).

This does NOT stack Module 4 on top of Module 2 -- it swaps the selector
used inside the same A1+2+3 pipeline (Module 1 substrate + Module 3
proactive triggering), replacing "Module 2's TOPSIS ranking decides which
band" with "Module 4's trained greedy Q-policy decides which band". This
directly answers whether the learned selector beats the rule-based one when
both are embedded in the same closed-loop pipeline, which is the natural
follow-up to Table IX's isolated DQN-vs-heuristics comparison.

Training uses the exact same procedure as Table IX's compare_dqn_vs_heuristics
in run_experiments.py: one DQN, seed=42, trained for 200 episodes at
n_ues=10/max_steps=30, eps_decay derived from the training budget so
exploration tapers across ~80% of training rather than collapsing in the
first couple episodes. The SAME trained policy (frozen, eps=0 / pure greedy)
is then evaluated -- embedded in the ablation's own 150-UE / 150-tick / 20-seed
pipeline -- exactly as A1+2+3's TOPSIS selector was.
"""
import random
import time as _time
import numpy as np

from multiband_planning import (
    HierarchicalMultiBandPlanner, Band, Position, UserEquipment,
    compute_network_kpis, PropagationModel,
)
from dqn_traffic_steering import (
    MultiBandRANEnv, EnvConfig, TrainingConfig, DQNTrainer, RewardWeights, TORCH_AVAILABLE,
    BAND_LIST, N_BANDS,
)

if TORCH_AVAILABLE:
    import torch
from predictive_handoff import PredictiveHandoffEngine, Obstacle
from context_aware_handoff import AppType

from ablation_experiment import (
    SEEDS, N_UES, N_TICKS, TICK_S, AREA_M, OBSTACLES,
    _build_population, _move, _recompute_throughput_frozen, _reassign,
    _snapshot_kpis, _avg_kpis, paired_ttest,
)


def train_dqn_like_table_ix(seed: int = 42, train_episodes: int = 200):
    """Mirrors run_experiments.compare_dqn_vs_heuristics's training block
    exactly (same env_cfg, same eps_decay derivation, same agent_kwargs)."""
    env_cfg = EnvConfig(n_ues=10, max_steps=30, reward_weights=RewardWeights())
    total_steps = train_episodes * env_cfg.n_ues * env_cfg.max_steps
    target_steps = max(int(0.8 * total_steps), 1)
    eps_decay = (0.05 / 1.0) ** (1.0 / target_steps)

    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        # BUG FIX (reproducibility-package audit, item 5): see run_experiments.py's
        # compare_dqn_vs_heuristics for the full explanation. This function mirrors
        # that one exactly and had the same gap.
        torch.manual_seed(seed)
    trainer = DQNTrainer(
        env_cfg=env_cfg,
        train_cfg=TrainingConfig(n_episodes=train_episodes,
                                  log_interval=train_episodes + 1,
                                  eval_interval=train_episodes + 1),
        agent_kwargs=dict(lr=1e-4, eps_decay=eps_decay, batch_size=32, hidden=64,
                           numpy_backend="mlp"),
    )
    print(f"Training DQN (Table-IX procedure): {train_episodes} episodes, seed={seed}...")
    trainer.train(verbose=False)
    print(f"Done. Final training eps: {trainer.agent.eps:.3f}")
    return trainer.agent


def _observe_ue_standalone(prop: PropagationModel, planner, ue, app_type_idx: int) -> np.ndarray:
    """Reimplements MultiBandRANEnv._observe_ue's exact feature layout
    (9 dims: 3 SNR + 3 load + velocity + app_type + bias), but against the
    ablation's own planner/population instead of a fresh MultiBandRANEnv."""
    snrs, loads = [], []
    for band in BAND_LIST:
        band_bss = [b for b in planner.base_stations if b.band == band]
        if band_bss:
            best_snr = max(prop.snr_db(bs, ue) for bs in band_bss)
            avg_load = float(np.mean([bs.load for bs in band_bss]))
        else:
            best_snr, avg_load = -99.0, 0.0
        snrs.append(np.clip((best_snr + 20) / 60, 0, 1))
        loads.append(avg_load)
    vel_norm = np.clip(ue.velocity / 50.0, 0, 1)
    app_norm = app_type_idx / 4.0
    return np.array(snrs + loads + [vel_norm, app_norm, 1.0], dtype=np.float32)


def run_config_A1_2_3_dqn(seed, agent, prop):
    """Same shape as ablation_experiment.run_config_A1_2_3, except: wherever
    that function calls selector.select_band()/scorer.score() (Module 2's
    TOPSIS ranking) to decide WHICH band, this calls agent._greedy_action()
    (Module 4's frozen trained policy) instead. Module 3's proactive
    triggering (WHEN to act) is unchanged."""
    planner, ues, app_types_enum = _build_population(seed)
    app_type_idx = [list(AppType).index(a) for a in app_types_enum]

    engine = PredictiveHandoffEngine(planner.base_stations)
    for ue in ues:
        engine.register_ue(ue)
    for obs in OBSTACLES:
        engine.add_obstacle(obs)

    rng = np.random.default_rng(seed + 40_000)
    snaps, ho_count, proactive_count = [], 0, 0
    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app_idx in zip(ues, app_type_idx):
            _move(ue, rng)
            obs_vec = _observe_ue_standalone(prop, planner, ue, app_idx)

            proactive_decision = engine.tick(ue, time_ms=t_ms)
            if proactive_decision is not None:
                action = agent._greedy_action(obs_vec)
                target = BAND_LIST[action]
                if target != ue.assigned_band:
                    if _reassign(planner, ue, target):
                        ho_count += 1
                        proactive_count += 1
                        continue
                _recompute_throughput_frozen(planner, ue)
                continue

            # Reactive step: Module 4's greedy policy picks the band every
            # tick (mirroring Module 2's reactive TOPSIS scoring cadence in
            # A1+2/A1+2+3), and a handover is only counted/applied when the
            # policy's chosen band differs from the current one.
            action = agent._greedy_action(obs_vec)
            target = BAND_LIST[action]
            if target != ue.assigned_band or ue.assigned_bs is None:
                if _reassign(planner, ue, target):
                    ho_count += 1
            else:
                _recompute_throughput_frozen(planner, ue)
        snaps.append(_snapshot_kpis(planner, ues))
    out = _avg_kpis(snaps)
    out["handover_count"] = ho_count
    out["proactive_handover_count"] = proactive_count
    return out


def main():
    agent = train_dqn_like_table_ix()
    prop = PropagationModel()

    results = []
    t0 = _time.time()
    for seed in SEEDS:
        results.append(run_config_A1_2_3_dqn(seed, agent, prop))
        print(f"seed {seed} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct",
               "handover_count"]

    print("\n" + "=" * 60)
    print("A1+2+3 with Module 4 (DQN) substituted for Module 2 (TOPSIS)")
    for m in metrics:
        vals = [r.get(m, 0.0) for r in results]
        print(f"  {m:<26} {np.mean(vals):>12.3f} ±{np.std(vals):<6.3f}")
    print("=" * 60)

    import json
    with open("ablation_row4_torch_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compare against the already-computed A1+2+3 (TOPSIS) row, if available.
    try:
        with open("ablation_results.json") as f:
            existing = json.load(f)
        a123 = existing["A1+2+3"]
        print("\nPaired t-test vs. A1+2+3 (TOPSIS), same 20 seeds:")
        for m in metrics:
            a = [r.get(m, 0.0) for r in a123]
            b = [r.get(m, 0.0) for r in results]
            t, p, d = paired_ttest(a, b)
            print(f"  {m:<24} TOPSIS vs DQN: t={t:.3f}  p={p:.4g}  d={d:.3f}")
    except FileNotFoundError:
        print("\n(ablation_results.json not found yet -- run ablation_experiment.py "
              "first for the TOPSIS-row comparison.)")


if __name__ == "__main__":
    main()
