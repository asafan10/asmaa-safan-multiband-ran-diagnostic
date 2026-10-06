"""
Ablation row 7 (follow-up to row 6): row 6 showed that a *band-level*
destination-load criterion added to TOPSIS does NOT fix the congestion
stampede from row 5 -- it makes P5 throughput and QoS coverage slightly
WORSE, not better (see ablation_row6_results.json and the paired
t-tests against row 5 printed by that script). This script tests why,
and whether a more targeted fix works.

Diagnosis: TOPSIS (with or without a load criterion) only ever chooses
a BAND (Sub-6 / mmWave / THz). Which CELL within that band a UE
connects to is decided entirely separately, by `_reassign()`, which
always picks the max-SNR cell in the target band and ignores load
completely. Because nearby UEs sharing similar propagation geometry
compute nearly identical per-band SNR rankings, they don't just agree
on which BAND is best (which row 6's band-level load term can
influence) -- they agree on which CELL within that band is best, and
_reassign always sends all of them to that same cell. A band-level
load criterion cannot fix a cell-level pileup.

This script keeps row 6's patches (score-only A3 trigger + band-level
load criterion in TOPSIS) and ADDS a load-aware cell-selection rule:
instead of always connecting to the max-SNR cell in the chosen band,
connect to the least-loaded cell among those within HYSTERESIS_MARGIN_DB
of the best SNR in that band (the same 3 dB margin already used
elsewhere in this paper's A3 hysteresis condition, not a new arbitrary
constant). This directly targets the actual pileup mechanism rather
than a proxy for it.
"""
import time as _time
import numpy as np

import context_aware_handoff as _cah
from context_aware_handoff import HandoffStateMachine, TOPSISBandScorer
import ablation_experiment as ae
from ablation_experiment import SEEDS, N_TICKS, run_config_A1_2, run_config_A1_2_3, paired_ttest
from ablation_row6_loadaware import (
    _patched_build_context, _patched_score, _patched_evaluate, EXTENDED_WEIGHTS,
)

HYSTERESIS_MARGIN_DB = 3.0   # same margin as the paper's own A3 condition


def _patched_reassign(planner, ue, band):
    """Load-aware cell selection: among cells in `band` within
    HYSTERESIS_MARGIN_DB of the best SNR, connect to the least-loaded
    one (ties broken by SNR), instead of always the single max-SNR
    cell. Falls back to the exact original behavior when there is only
    one acceptable candidate."""
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
    best_bs = min(acceptable, key=lambda b: (b.load, -snrs[b.bs_id]))

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
    ae._reassign = _patched_reassign

    configs = [("A1+2 (score-trigger+load+cell)", run_config_A1_2),
               ("A1+2+3 (score-trigger+load+cell)", run_config_A1_2_3)]
    results = {name: [] for name, _ in configs}

    planner, ues, app_types = ae._build_population(SEEDS[0])
    selector = _cah.ContextAwareBandSelector(weight_mode="static")
    rng = np.random.default_rng(SEEDS[0] + 20_000)
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
    print(f"Peak single-BS active-UE count on seed {SEEDS[0]} (score-trigger+load+cell variant): {peak_load}")

    t0 = _time.time()
    for seed in SEEDS:
        for name, fn in configs:
            results[name].append(fn(seed))
        print(f"seed {seed} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct", "handover_count"]

    print("\n" + "=" * 90)
    print(f"{'Metric':<26}" + "".join(f"{name:>32}" for name, _ in configs))
    for m in metrics:
        row = f"{m:<26}"
        for name, _ in configs:
            vals = [r.get(m, 0.0) for r in results[name]]
            row += f"{np.mean(vals):>24.3f} ±{np.std(vals):<6.2f}"
        print(row)
    print("=" * 90)

    import json
    with open("ablation_row7_results.json", "w") as f:
        json.dump({"peak_load": peak_load, "results": results}, f, indent=2)

    comparisons = [
        ("original SNR-gated", "ablation_results.json", None,
         [("A1+2", "A1+2 (score-trigger+load+cell)"), ("A1+2+3", "A1+2+3 (score-trigger+load+cell)")]),
        ("score-trigger only (row 5, no load term)", "ablation_row5_results.json", "results",
         [("A1+2 (score-trigger)", "A1+2 (score-trigger+load+cell)"),
          ("A1+2+3 (score-trigger)", "A1+2+3 (score-trigger+load+cell)")]),
        ("band-level load only (row 6, no cell fix)", "ablation_row6_results.json", "results",
         [("A1+2 (score-trigger+load)", "A1+2 (score-trigger+load+cell)"),
          ("A1+2+3 (score-trigger+load)", "A1+2+3 (score-trigger+load+cell)")]),
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
