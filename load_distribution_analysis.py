"""
load_distribution_analysis.py — per-cell load distribution analysis
=====================================================================
Resolves the Module 1 load-balance caveat flagged by two review passes:
Table V.a/b reports the proposed planner has a HIGHER load std-dev than
BL-1 (Max-SNR) (0.10+/-0.02 vs 0.08+/-0.00, p=5.94e-04 at n_ues=150;
p=2.27e-05 at a second scale), and the paper explicitly declines to say
whether that higher variance means load is MORE CONCENTRATED (a few
overloaded cells -- bad) or MORE SPREAD (deliberately shedding load
toward underused capacity -- the intended behavior, and arguably good).//
Aggregate std-dev cannot distinguish these two cases; the full per-cell
distribution can.

This script re-runs the exact same comparison compare_planners() already
uses (Proposed vs. BL-1, COMPARISON_N_UES=150, the canonical 20-seed
list) but instead of collapsing to load_std, it retains every individual
base station's `bs.load` value across all seeds, and reports:
  - the full pooled per-cell load distribution (for a CDF/boxplot figure)
  - overloaded-cell rate (bs.load >= 1.0) -- already in the paper, kept
    here as a cross-check
  - a NEW "high-load" tail metric: % of cells with load >= 0.8 (loaded
    but not yet in violation) and >= 0.9, since if the proposed method's
    extra variance comes from pushing a few cells right up to the
    capacity ceiling (without tipping them over), that is the
    "concentration" reading; if instead its extra variance comes from
    spreading load further down into idle cells (more cells in the
    0.0-0.3 range than BL-1), that is the "spreading" reading.
"""

import json
import numpy as np

from run_experiments import SEEDS, COMPARISON_N_UES, _run_one_seed
from multiband_planning import (
    HierarchicalMultiBandPlanner, MaxSNRPlanner, build_scenario, clone_topology,
)


def collect_per_cell_loads(seeds=SEEDS, n_ues=COMPARISON_N_UES):
    """Re-runs the identical proposed-vs-BL-1 comparison compare_planners()
    uses, but returns the full per-BS load array for every seed instead of
    just the aggregate std-dev."""
    proposed_loads_by_seed = []
    bl1_loads_by_seed = []

    for seed in seeds:
        base_stations, ues = build_scenario(seed=seed, n_ues=n_ues)

        proposed = HierarchicalMultiBandPlanner()
        proposed.base_stations = base_stations
        proposed.ues = ues
        proposed._bs_index = {b.bs_id: b for b in base_stations}
        proposed.run_planning()
        proposed_loads_by_seed.append([bs.load for bs in proposed.base_stations])

        bl1_bss, bl1_ues = clone_topology(base_stations, ues)
        bl1 = MaxSNRPlanner(bl1_bss)
        bl1.ues = bl1_ues
        bl1.run_planning()
        bl1_loads_by_seed.append([bs.load for bs in bl1.base_stations])

    return proposed_loads_by_seed, bl1_loads_by_seed


def summarize(loads_by_seed, label):
    pooled = np.concatenate([np.asarray(l) for l in loads_by_seed])
    n_cells_per_seed = len(loads_by_seed[0])
    result = {
        "label": label,
        "n_seeds": len(loads_by_seed),
        "n_cells_per_seed": n_cells_per_seed,
        "n_pooled_observations": int(pooled.size),
        "mean": float(pooled.mean()),
        "std": float(pooled.std()),
        "min": float(pooled.min()),
        "p5": float(np.percentile(pooled, 5)),
        "p25": float(np.percentile(pooled, 25)),
        "median": float(np.percentile(pooled, 50)),
        "p75": float(np.percentile(pooled, 75)),
        "p95": float(np.percentile(pooled, 95)),
        "max": float(pooled.max()),
        "pct_overloaded_ge_1.0": float(100 * np.mean(pooled >= 1.0)),
        "pct_high_load_ge_0.9": float(100 * np.mean(pooled >= 0.9)),
        "pct_high_load_ge_0.8": float(100 * np.mean(pooled >= 0.8)),
        "pct_idle_le_0.1": float(100 * np.mean(pooled <= 0.1)),
        "pct_idle_le_0.3": float(100 * np.mean(pooled <= 0.3)),
    }
    return result, pooled


def main():
    print("=" * 72)
    print("  PER-CELL LOAD DISTRIBUTION ANALYSIS: Proposed vs. BL-1 (Max-SNR)")
    print(f"  seeds = {SEEDS}")
    print(f"  n_ues = {COMPARISON_N_UES} (same scale as Table V.a/b)")
    print("=" * 72)

    proposed_loads, bl1_loads = collect_per_cell_loads()

    proposed_summary, proposed_pooled = summarize(proposed_loads, "Proposed")
    bl1_summary, bl1_pooled = summarize(bl1_loads, "BL-1 (Max-SNR)")

    for s in (proposed_summary, bl1_summary):
        print(f"\n  --- {s['label']} ---")
        print(f"  pooled cells (n_seeds x n_cells) = {s['n_pooled_observations']}")
        print(f"  mean={s['mean']:.4f}  std={s['std']:.4f}")
        print(f"  min={s['min']:.4f}  p5={s['p5']:.4f}  p25={s['p25']:.4f}  "
              f"median={s['median']:.4f}  p75={s['p75']:.4f}  p95={s['p95']:.4f}  max={s['max']:.4f}")
        print(f"  overloaded (>=1.0): {s['pct_overloaded_ge_1.0']:.2f}%   "
              f"high-load (>=0.9): {s['pct_high_load_ge_0.9']:.2f}%   "
              f"high-load (>=0.8): {s['pct_high_load_ge_0.8']:.2f}%")
        print(f"  idle (<=0.1): {s['pct_idle_le_0.1']:.2f}%   idle (<=0.3): {s['pct_idle_le_0.3']:.2f}%")

    out = {
        "seeds": SEEDS,
        "n_ues": COMPARISON_N_UES,
        "proposed": proposed_summary,
        "bl1": bl1_summary,
        "proposed_pooled_loads": proposed_pooled.tolist(),
        "bl1_pooled_loads": bl1_pooled.tolist(),
    }
    with open("load_distribution_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: load_distribution_results.json")

    # Interpretation hint (not a claim -- the paper text will state the
    # actual reading based on these numbers, verified, not assumed).
    print("\n" + "=" * 72)
    print("  INTERPRETATION GUIDE (verify against numbers above, don't assume):")
    print("  - If Proposed has a FATTER upper tail (higher p95/max, higher")
    print("    %>=0.9) than BL-1 WITHOUT a correspondingly fatter lower tail,")
    print("    the extra variance is CONCENTRATION (a few cells pushed hard).")
    print("  - If Proposed has BOTH a fatter upper tail AND a fatter lower")
    print("    tail (more idle cells too) than BL-1, the extra variance is")
    print("    SPREADING (deliberately offloading toward idle capacity),")
    print("    which is a materially different, more favorable story.")
    print("=" * 72)


if __name__ == "__main__":
    main()
