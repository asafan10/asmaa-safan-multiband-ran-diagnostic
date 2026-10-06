"""
Ablation row 8 (follow-up to row 7): row 7's deterministic
least-loaded-cell tie-break did NOT fix the congestion stampede either
(P5 throughput ~0.23 Mbps, statistically indistinguishable from row 6).
Root cause, confirmed by inspection: because every UE observes the same
per-tick load snapshot and all UEs move/reselect synchronously within a
tick, a DETERMINISTIC least-loaded rule just relocates the herd -- many
UEs that all saw cell A as least-loaded in tick t simultaneously pick
cell A in tick t, overloading it in near-synchrony, exactly the
"thundering herd" failure mode of naive greedy load balancing in any
decentralized system with synchronized decisions.

This script tests the standard fix for that specific failure mode:
randomized "power of two choices" (Mitzenmacher, 2001) -- among the
candidate cells within the same SNR-acceptability margin used in row 7
(HYSTERESIS_MARGIN_DB), sample two at random and connect to whichever
of the two has lower current load, rather than deterministically
picking the single least-loaded one. This breaks the synchronization
that caused row 7 to fail, at the cost of no longer guaranteeing the
single best cell is chosen.
"""
import time as _time
import random
import numpy as np

import context_aware_handoff as _cah
from context_aware_handoff import HandoffStateMachine, TOPSISBandScorer
import ablation_experiment as ae
from ablation_experiment import SEEDS, N_TICKS, run_config_A1_2, run_config_A1_2_3, paired_ttest
from ablation_row6_loadaware import _patched_build_context, _patched_score, _patched_evaluate
from ablation_row7_cellload import HYSTERESIS_MARGIN_DB


def _patched_reassign_power2(planner, ue, band):
    if ue.assigned_bs is not None and ue.assigned_bs in planner._bs_index:
        old_bs = planner._bs_index[ue.assigned_bs]
        if ue.ue_id in old_bs.active_ues:
            old_bs.active_ues.remove(ue.ue_id)
        ue.assigned_bs, ue.assigned_band, ue.throughput_mbps = None, None, 0.0

    candidates = [bs for bs in planner.base_stations if bs.band == band]
    if not candidates:
        return False

    snrs = {bs.bs_id: planner.prop_model.snr_db(bs, ue) for bs in candidates}
    best_snr = max(snrs.values())
    acceptable = [bs for bs in candidates if snrs[bs.bs_id] >= best_snr - HYSTERESIS_MARGIN_DB]

    if len(acceptable) >= 2:
        a, b = random.sample(acceptable, 2)
        best_bs = a if a.load <= b.load else b
    else:
        best_bs = acceptable[0]

    best_bs.active_ues.append(ue.ue_id)
    ue.assigned_bs = best_bs.bs_id
    ue.assigned_band = band
    n = max(len(best_bs.active_ues), 1)
    ue.throughput_mbps = planner.prop_model.shannon_capacity_mbps(best_bs, ue) / n
    return True


def main():
    _orig_build_context = _cah.ContextAwareBandSelector.build_context
    _patched_build_context.__wrapped_orig__ = _orig_build_context
    _cah.ContextAwareBandSelector.build_context = _patched_build_context
    TOPSISBandScorer.score = _patched_score
    HandoffStateMachine.evaluate = _patched_evaluate
    ae._reassign = _patched_reassign_power2

    configs = [("A1+2 (score-trigger+power2)", run_config_A1_2),
               ("A1+2+3 (score-trigger+power2)", run_config_A1_2_3)]
    results = {name: [] for name, _ in configs}

    planner, ues, app_types = ae._build_population(SEEDS[0])
    selector = _cah.ContextAwareBandSelector(weight_mode="static")
    rng = np.random.default_rng(SEEDS[0] + 20_000)
    random.seed(SEEDS[0])
    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app in zip(ues, app_types):
            ae._move(ue, rng)
            ctx = selector.build_context(ue, app, planner.base_stations)
            b, ho_event = selector.select_band(ctx, time_ms=t_ms)
            if b is not None and (ue.assigned_band != b or ue.assigned_bs is None):
                ae._reassign(planner, ue, b)
            else:
                ae._recompute_throughput_frozen(planner, ue)
    peak_load = max(len(bs.active_ues) for bs in planner.base_stations)
    print(f"Peak single-BS active-UE count on seed {SEEDS[0]} (power-of-2-choices variant): {peak_load}")

    t0 = _time.time()
    for seed in SEEDS:
        random.seed(seed)
        for name, fn in configs:
            results[name].append(fn(seed))
        print(f"seed {seed} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct", "handover_count"]

    print("\n" + "=" * 90)
    print(f"{'Metric':<26}" + "".join(f"{name:>30}" for name, _ in configs))
    for m in metrics:
        row = f"{m:<26}"
        for name, _ in configs:
            vals = [r.get(m, 0.0) for r in results[name]]
            row += f"{np.mean(vals):>22.3f} ±{np.std(vals):<6.2f}"
        print(row)
    print("=" * 90)

    import json
    with open("ablation_row8_results.json", "w") as f:
        json.dump({"peak_load": peak_load, "results": results}, f, indent=2)

    comparisons = [
        ("original SNR-gated", "ablation_results.json", None,
         [("A1+2", "A1+2 (score-trigger+power2)"), ("A1+2+3", "A1+2+3 (score-trigger+power2)")]),
        ("score-trigger only (row 5)", "ablation_row5_results.json", "results",
         [("A1+2 (score-trigger)", "A1+2 (score-trigger+power2)"),
          ("A1+2+3 (score-trigger)", "A1+2+3 (score-trigger+power2)")]),
    ]
    for label, path, nested_key, pairs in comparisons:
        try:
            with open(path) as f:
                other = json.load(f)
            other_results = other[nested_key] if nested_key else other
            print(f"\nPaired t-test vs. {label} (same 20 seeds):")
            for orig_name, new_name in pairs:
                if orig_name not in other_results:
                    print(f"  ({orig_name} not found in {path}, skipping)")
                    continue
                print(f"  -- {orig_name} vs {new_name} --")
                for m in metrics:
                    a = [r.get(m, 0.0) for r in other_results[orig_name]]
                    b = [r.get(m, 0.0) for r in results[new_name]]
                    t, p, d = paired_ttest(a, b)
                    print(f"    {m:<24} t={t:.3f}  p={p:.4g}  d={d:.3f}")
        except FileNotFoundError:
            print(f"\n({path} not found -- skip comparison)")


if __name__ == "__main__":
    main()
