"""
Cooldown-gated variant of ablation_row4.run_config_A1_2_3_dqn, testing the
paper's own "distribution-shift and hysteresis gap" attribution for Module 4's
severe reselection instability in the closed-loop K=150 ablation (Table VIII
row 4), rather than leaving it asserted.

Mechanism: identical to run_config_A1_2_3_dqn (real dueling/double-DQN policy
substituted for Module 2's TOPSIS ranking inside the A1+2+3 pipeline, Module
3's proactive trigger unchanged), with ONE addition -- a fixed per-UE cooldown
counter. After any reselection (proactive or reactive), that UE cannot be
reselected again for `cooldown_ticks` ticks; the policy is not re-queried for
that UE while its cooldown is active, and its current band/throughput is kept.
This is deliberately the simplest possible hysteresis mechanism (no learning,
no adaptation), to test whether ANY decoupled-from-SNR hysteresis closes the
gap toward Module 2's reselection range, or whether something else is going on.
"""
import random
import time as _time
import numpy as np

from multiband_planning import PropagationModel
from predictive_handoff import PredictiveHandoffEngine, Obstacle
from context_aware_handoff import AppType
from dqn_traffic_steering import BAND_LIST

from ablation_experiment import (
    SEEDS, N_TICKS, OBSTACLES,
    _build_population, _move, _recompute_throughput_frozen, _reassign,
    _snapshot_kpis, _avg_kpis, paired_ttest,
)
from ablation_row4_torch_v2 import _observe_ue_standalone, train_dqn_like_table_ix


def run_config_A1_2_3_dqn_cooldown(seed, agent, prop, cooldown_ticks: int):
    planner, ues, app_types_enum = _build_population(seed)
    app_type_idx = [list(AppType).index(a) for a in app_types_enum]

    engine = PredictiveHandoffEngine(planner.base_stations)
    for ue in ues:
        engine.register_ue(ue)
    for obs in OBSTACLES:
        engine.add_obstacle(obs)

    rng = np.random.default_rng(seed + 40_000)
    snaps, ho_count, proactive_count = [], 0, 0
    cooldown = {id(ue): 0 for ue in ues}

    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app_idx in zip(ues, app_type_idx):
            _move(ue, rng)
            uid = id(ue)
            if cooldown[uid] > 0:
                cooldown[uid] -= 1
                _recompute_throughput_frozen(planner, ue)
                continue

            obs_vec = _observe_ue_standalone(prop, planner, ue, app_idx)
            proactive_decision = engine.tick(ue, time_ms=t_ms)
            if proactive_decision is not None:
                action = agent._greedy_action(obs_vec)
                target = BAND_LIST[action]
                if target != ue.assigned_band:
                    if _reassign(planner, ue, target):
                        ho_count += 1
                        proactive_count += 1
                        cooldown[uid] = cooldown_ticks
                        continue
                _recompute_throughput_frozen(planner, ue)
                continue

            action = agent._greedy_action(obs_vec)
            target = BAND_LIST[action]
            if target != ue.assigned_band or ue.assigned_bs is None:
                if _reassign(planner, ue, target):
                    ho_count += 1
                    cooldown[uid] = cooldown_ticks
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

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct",
               "handover_count"]

    import json
    with open("ablation_row4_torch_v2_results.json") as f:
        no_cooldown = json.load(f)
    with open("ablation_results.json") as f:
        existing = json.load(f)
    a123 = existing["A1+2+3"]

    all_results = {}
    for cd in [5, 10, 20]:
        t0 = _time.time()
        results = []
        for seed in SEEDS:
            results.append(run_config_A1_2_3_dqn_cooldown(seed, agent, prop, cd))
        print(f"cooldown={cd} ticks done ({_time.time()-t0:.1f}s)")
        all_results[cd] = results

        print(f"\n=== Cooldown = {cd} ticks ===")
        for m in metrics:
            vals = [r.get(m, 0.0) for r in results]
            print(f"  {m:<26} {np.mean(vals):>12.3f} +/-{np.std(vals):<6.3f}")

        print(f"Paired t-test: no-cooldown DQN vs cooldown={cd}:")
        for m in metrics:
            a = [r.get(m, 0.0) for r in no_cooldown]
            b = [r.get(m, 0.0) for r in results]
            t, p, d = paired_ttest(a, b)
            print(f"  {m:<24} t={t:.3f}  p={p:.4g}  d={d:.3f}")

        print(f"Paired t-test: TOPSIS (A1+2+3) vs cooldown={cd}:")
        for m in metrics:
            a = [r.get(m, 0.0) for r in a123]
            b = [r.get(m, 0.0) for r in results]
            t, p, d = paired_ttest(a, b)
            print(f"  {m:<24} t={t:.3f}  p={p:.4g}  d={d:.3f}")

    with open("ablation_row4_cooldown_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
