"""
STATUS: UNUSED / ABANDONED. Not referenced by run_experiments.py,
ablation_experiment.py, or any other script in this package, and no
result in the current paper comes from this file. Its own comments
below refer to "Table IX" as a DQN-vs-heuristics table -- that is not
what Table IX is in the current paper (it is the destination-load-aware
TOPSIS fix-variant table), confirming this script predates the paper's
later restructuring and was never updated or reconnected. Kept for
provenance only. The current paper's real-architecture DQN results come
from dqn_traffic_steering.py (_QNetwork + DQNAgent, PyTorch path).

SOTA-architecture comparison for Module 4 (paper item #5): does a Dueling
Double-DQN -- the architectural approach used by [14] Liang et al. (2026)
and [15] He et al. (2017), both DRL-for-HetNet-RRM works already cited in
Table I -- outperform this paper's existing single-stream DQN, when BOTH
are trained under IDENTICAL conditions in this paper's own environment
(same reward, same state/action space, same episode budget, same eval
seeds)? This is explicitly NOT a reproduction of [14]/[15]'s own reported
numbers (different topology/reward/physics model, not reproducible
without their code) -- see the three pre-commitments recorded before this
script was written (conversation record, immediately preceding this file).

Everything except the Q-function architecture and the double-DQN target
computation is held fixed at Table IX's own values, per pre-commitment #1:
env_cfg=(n_ues=10, max_steps=30), train_episodes=200, lr=1e-4,
batch_size=32, hidden=64 (encoder capacity, matching the existing
single-stream MLP's hidden width), eps_decay derived from the training
budget exactly as compare_dqn_vs_heuristics() already does, seed=42 for
training, 20 eval seeds (9000..9019) for evaluation -- the same eval
seeds Table IX itself uses (see README.md / SEED_MANIFEST.md, item #4).
"""
import random
import time as _time
import numpy as np
from scipy import stats

from dqn_traffic_steering import (
    EnvConfig, TrainingConfig, DQNTrainer, RewardWeights, MultiBandRANEnv,
)
from multiband_planning import Band
from dueling_ddqn_agent import DuelingDoubleDQNAgent
from run_experiments import (
    _make_dqn_policy, _run_episode_with_policy, _policy_always_band,
    _policy_random, _policy_bl2, SEEDS,
)

TRAIN_EPISODES = 200
EVAL_EPISODES = 20          # matches Table IX's (corrected) eval-seed count
TRAIN_SEED = 42             # matches Table IX's training seed
EVAL_SEEDS = [9000 + i for i in range(EVAL_EPISODES)]  # identical to Table IX


def _eps_decay_for(train_episodes, env_cfg):
    total_steps = train_episodes * env_cfg.n_ues * env_cfg.max_steps
    target_steps = max(int(0.8 * total_steps), 1)
    return (0.05 / 1.0) ** (1.0 / target_steps)


def train_dueling_ddqn(seed: int = TRAIN_SEED, train_episodes: int = TRAIN_EPISODES):
    """Identical procedure to compare_dqn_vs_heuristics()'s own DQN
    training call, with ONLY the agent class swapped in."""
    env_cfg = EnvConfig(n_ues=10, max_steps=30, reward_weights=RewardWeights())
    eps_decay = _eps_decay_for(train_episodes, env_cfg)

    random.seed(seed)
    np.random.seed(seed)
    trainer = DQNTrainer(
        env_cfg=env_cfg,
        train_cfg=TrainingConfig(n_episodes=train_episodes,
                                  log_interval=train_episodes + 1,
                                  eval_interval=train_episodes + 1),
        agent_kwargs=dict(),  # unused -- agent replaced below before construction completes
    )
    # DQNTrainer's constructor always builds a DQNAgent internally; replace
    # it with the dueling-double agent, built with the SAME hyperparameters
    # (lr, batch_size, hidden, eps_decay) Table IX's own DQN uses, per
    # pre-commitment #1. This is the only line where the two training runs
    # actually differ.
    trainer.agent = DuelingDoubleDQNAgent(
        obs_dim=trainer.env.obs_dim, act_dim=trainer.env.act_dim,
        lr=1e-4, eps_decay=eps_decay, batch_size=32, hidden=64,
    )

    print(f"Training Dueling-Double-DQN: seed={seed}, {train_episodes} episodes, "
          f"eps_decay={eps_decay:.6f}...")
    t0 = _time.time()
    trainer.train(verbose=False)
    print(f"Done in {_time.time()-t0:.1f}s. Final training eps: {trainer.agent.eps:.3f}")
    return trainer.agent, env_cfg


def evaluate(agent, env_cfg, eval_seeds=EVAL_SEEDS):
    policy_fn = _make_dqn_policy(agent)
    tputs, rewards, entropies = [], [], []
    for s in eval_seeds:
        # Reseed BEFORE constructing the env, not after -- MultiBandRANEnv.__init__
        # places base stations via deploy_grid()'s random.uniform() calls at
        # construction time, so seeding afterward only pins UE state (set in
        # .reset()) and silently leaves BS topology dependent on execution
        # history. See run_experiments.py's compare_dqn_vs_heuristics for the
        # full diagnosis (same bug, same fix, found during this comparison).
        random.seed(s); np.random.seed(s)
        env = MultiBandRANEnv(env_cfg)
        ep = _run_episode_with_policy(env, policy_fn, seed=s)
        tputs.append(ep["avg_throughput_mbps"])
        rewards.append(ep["total_reward"])
        entropies.append(ep["band_entropy"])
    return {
        "avg_throughput_mbps": tputs, "total_reward": rewards, "band_entropy": entropies,
    }


def paired_compare(name_a, a_vals, name_b, b_vals):
    a, b = np.array(a_vals), np.array(b_vals)
    diff = a - b
    if np.allclose(diff, diff[0]) and np.isclose(diff.std(), 0):
        t, p, d = float("nan"), float("nan"), float("nan")
    else:
        t, p = stats.ttest_rel(a, b)
        d = float(diff.mean() / diff.std()) if diff.std() > 1e-12 else float("nan")
    print(f"  {name_a} vs {name_b}: t={t:.3f}  p={p:.4g}  d={d:.3f}  "
          f"({name_a} mean={a.mean():.2f}, {name_b} mean={b.mean():.2f})")
    return {"t": float(t), "p": float(p), "d": d, "a_mean": float(a.mean()), "b_mean": float(b.mean())}


def main():
    # 1. Train and evaluate the single-stream DQN exactly as Table IX does
    #    (re-run here, not read from a stored file, so both agents are
    #    compared from runs produced in the same session/process).
    env_cfg = EnvConfig(n_ues=10, max_steps=30, reward_weights=RewardWeights())
    eps_decay = _eps_decay_for(TRAIN_EPISODES, env_cfg)
    random.seed(TRAIN_SEED); np.random.seed(TRAIN_SEED)
    baseline_trainer = DQNTrainer(
        env_cfg=env_cfg,
        train_cfg=TrainingConfig(n_episodes=TRAIN_EPISODES,
                                  log_interval=TRAIN_EPISODES + 1,
                                  eval_interval=TRAIN_EPISODES + 1),
        agent_kwargs=dict(lr=1e-4, eps_decay=eps_decay, batch_size=32, hidden=64,
                           numpy_backend="mlp"),
    )
    print(f"Training single-stream DQN (Table IX procedure): seed={TRAIN_SEED}, "
          f"{TRAIN_EPISODES} episodes...")
    t0 = _time.time()
    baseline_trainer.train(verbose=False)
    print(f"Done in {_time.time()-t0:.1f}s.")
    baseline_eval = evaluate(baseline_trainer.agent, env_cfg)

    # 2. Train and evaluate the dueling-double-DQN variant, identical
    #    conditions otherwise.
    dueling_agent, _ = train_dueling_ddqn()
    dueling_eval = evaluate(dueling_agent, env_cfg)

    # 3. Heuristic baselines, evaluated identically (same seeds/env), for
    #    context -- not the primary comparison (pre-commitment #2).
    heuristic_policies = {
        "Always-mmWave": _policy_always_band(Band.MMWAVE),
        "Always-Sub6": _policy_always_band(Band.SUB6),
        "Random": _policy_random,
        "BL-2 (threshold)": _policy_bl2,
    }
    heuristic_eval = {}
    for name, fn in heuristic_policies.items():
        tputs = []
        for s in EVAL_SEEDS:
            random.seed(s); np.random.seed(s)  # seed BEFORE construction -- see evaluate()
            env = MultiBandRANEnv(env_cfg)
            ep = _run_episode_with_policy(env, fn, seed=s)
            tputs.append(ep["avg_throughput_mbps"])
        heuristic_eval[name] = tputs

    print("\n" + "=" * 78)
    print("  SOTA-ARCHITECTURE COMPARISON: single-stream DQN vs. Dueling-Double-DQN")
    print(f"  (both: seed={TRAIN_SEED} training, {TRAIN_EPISODES} episodes, "
          f"{EVAL_EPISODES} eval seeds, identical env/reward/hyperparameters)")
    print("=" * 78)
    all_series = {
        "Single-stream DQN (Table IX)": baseline_eval["avg_throughput_mbps"],
        "Dueling-Double-DQN":            dueling_eval["avg_throughput_mbps"],
        **heuristic_eval,
    }
    for name, vals in all_series.items():
        v = np.array(vals)
        print(f"  {name:<32} avg throughput: {v.mean():>10.2f} +/- {v.std():<10.2f} Mbps")

    print(f"\n  PRIMARY comparison (pre-committed metric: avg. throughput, paired t-test):")
    verdict = paired_compare(
        "Dueling-Double-DQN", dueling_eval["avg_throughput_mbps"],
        "Single-stream DQN", baseline_eval["avg_throughput_mbps"],
    )

    import json
    out = {
        "train_seed": TRAIN_SEED, "train_episodes": TRAIN_EPISODES,
        "eval_seeds": EVAL_SEEDS,
        "single_stream_dqn": baseline_eval,
        "dueling_double_dqn": dueling_eval,
        "heuristics": heuristic_eval,
        "primary_verdict_dueling_vs_single_stream": verdict,
    }
    with open("dueling_ddqn_sota_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to dueling_ddqn_sota_results.json")


if __name__ == "__main__":
    main()
