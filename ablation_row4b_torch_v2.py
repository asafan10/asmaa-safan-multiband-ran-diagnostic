"""
Ablation row 4b: does retraining the DQN policy AT the ablation's own
K=150 scale (rather than reusing the small-scale, n_ues=10 policy trained
for Table IX) resolve the severe reselection instability found when
Module 4 is substituted for Module 2 in the same A1+2+3 pipeline
(ablation_row4.py / paper Section IV.G)?

Training procedure: identical to run_experiments.compare_dqn_vs_heuristics
/ ablation_row4.train_dqn_like_table_ix (same eps_decay derivation, same
agent hyperparameters: lr=1e-4, batch_size=32, hidden=64, numpy MLP
backend), with ONE change -- env_cfg.n_ues=150 instead of 10 (max_steps
kept at 30, matching Table IX's own per-episode tick count, so only the
population size that differs). train_episodes is reduced from Table IX's
200 to 60 as a compute-budget concession, disclosed explicitly: 60
episodes at 150 UEs is still 4.5x Table IX's total training-step count,
but this is NOT the "same episode budget, bigger population" comparison
originally intended -- it is "smaller episode budget, larger step count,
bigger population," and is reported as such.

The retrained policy is then evaluated exactly as ablation_row4.py's
policy was: embedded in the same A1+2+3 pipeline (Module 1 substrate +
Module 3 proactive triggering, Module 4 replacing Module 2 as the
selector), across the same 20 ablation seeds, same 150 ticks, same KPIs.
"""
import random
import time as _time
import numpy as np

from dqn_traffic_steering import (
    EnvConfig, TrainingConfig, DQNTrainer, RewardWeights, TORCH_AVAILABLE,
)
if TORCH_AVAILABLE:
    import torch
from multiband_planning import PropagationModel

from ablation_experiment import SEEDS, paired_ttest
from ablation_row4_torch_v2 import run_config_A1_2_3_dqn


def train_dqn_at_k150(seed: int = 42, train_episodes: int = 60,
                       n_ues: int = 150, max_steps: int = 30):
    """Same procedure as ablation_row4.train_dqn_like_table_ix, but at the
    ablation's own population scale instead of Table IX's n_ues=10."""
    env_cfg = EnvConfig(n_ues=n_ues, max_steps=max_steps, reward_weights=RewardWeights())
    total_steps = train_episodes * env_cfg.n_ues * env_cfg.max_steps
    target_steps = max(int(0.8 * total_steps), 1)
    eps_decay = (0.05 / 1.0) ** (1.0 / target_steps)

    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        # BUG FIX (reproducibility-package audit, item 5): see
        # run_experiments.py's compare_dqn_vs_heuristics for the full explanation.
        torch.manual_seed(seed)
    trainer = DQNTrainer(
        env_cfg=env_cfg,
        train_cfg=TrainingConfig(n_episodes=train_episodes,
                                  log_interval=train_episodes + 1,
                                  eval_interval=train_episodes + 1),
        agent_kwargs=dict(lr=1e-4, eps_decay=eps_decay, batch_size=32, hidden=64,
                           numpy_backend="mlp"),
    )
    print(f"Training DQN at K={n_ues} (Table-IX procedure, same {train_episodes}-episode "
          f"budget, {total_steps} total steps): seed={seed}...")
    t0 = _time.time()
    trainer.train(verbose=False)
    print(f"Done in {_time.time()-t0:.1f}s. Final training eps: {trainer.agent.eps:.3f}")
    return trainer.agent


def main():
    agent = train_dqn_at_k150()
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
    print("A1+2+3 with Module 4 (DQN, RETRAINED AT K=150) substituted for Module 2")
    for m in metrics:
        vals = [r.get(m, 0.0) for r in results]
        print(f"  {m:<26} {np.mean(vals):>12.3f} ±{np.std(vals):<6.3f}")
    print("=" * 60)

    import json
    with open("ablation_row4b_torch_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compare against the small-scale-trained row4 policy AND the TOPSIS row.
    try:
        with open("ablation_row4_torch_v2_results.json") as f:
            small_scale = json.load(f)
        print("\nPaired t-test: RETRAINED-AT-K150 vs. SMALL-SCALE-TRAINED (Table IX policy), same 20 seeds:")
        for m in metrics:
            a = [r.get(m, 0.0) for r in small_scale]
            b = [r.get(m, 0.0) for r in results]
            t, p, d = paired_ttest(a, b)
            print(f"  {m:<24} small-scale vs retrained: t={t:.3f}  p={p:.4g}  d={d:.3f}")
    except FileNotFoundError:
        print("\n(ablation_row4_results.json not found -- skip small-scale comparison)")

    try:
        with open("ablation_results.json") as f:
            existing = json.load(f)
        a123 = existing["A1+2+3"]
        print("\nPaired t-test: RETRAINED-AT-K150 vs. TOPSIS (A1+2+3), same 20 seeds:")
        for m in metrics:
            a = [r.get(m, 0.0) for r in a123]
            b = [r.get(m, 0.0) for r in results]
            t, p, d = paired_ttest(a, b)
            print(f"  {m:<24} TOPSIS vs retrained-DQN: t={t:.3f}  p={p:.4g}  d={d:.3f}")
    except FileNotFoundError:
        print("\n(ablation_results.json not found -- skip TOPSIS comparison)")


if __name__ == "__main__":
    main()
