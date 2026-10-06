"""
w_load sensitivity check (reviewer round-5 concern #3): does the negative
result for the band-level load criterion depend on the specific weight
w_load = 0.15 used in the main run? Tested here at 0.05, 0.15 (reused from
the main run), 0.30, and 0.50, on a reduced 8-seed subset (first 8 of the
20) for computational tractability -- explicitly disclosed as such in the
paper. Uses the band-level-load-only patch (no cell-level fix), A1+2+3
configuration, same peak-load-as-running-max methodology as the main run.
"""
import time as _time
import json
import numpy as np
import random as _random

import context_aware_handoff as _cah
from context_aware_handoff import HandoffStateMachine, TOPSISBandScorer, Band
import ablation_experiment as ae
from ablation_experiment import SEEDS, N_TICKS
from table_x_full_stats import (
    _score_only_evaluate, make_load_aware_score, run_A1_2_3_with_peak,
    _ORIG_build_context, _ORIG_score, _ORIG_evaluate, _ORIG_reassign,
)

SUBSET_SEEDS = SEEDS[:8]


def reset():
    _cah.ContextAwareBandSelector.build_context = _ORIG_build_context
    TOPSISBandScorer.score = _ORIG_score
    HandoffStateMachine.evaluate = _ORIG_evaluate
    ae._reassign = _ORIG_reassign


def run_weight(w):
    reset()
    bc, sc, ext = make_load_aware_score(w)
    _cah.ContextAwareBandSelector.build_context = bc
    TOPSISBandScorer.score = sc
    HandoffStateMachine.evaluate = _score_only_evaluate
    results = []
    for seed in SUBSET_SEEDS:
        _random.seed(seed)
        results.append(run_A1_2_3_with_peak(seed))
    return results


def main():
    weights = [0.05, 0.30, 0.50]
    out = {}
    t0 = _time.time()
    for w in weights:
        res = run_weight(w)
        out[str(w)] = res
        print(f"w_load={w} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["qos_coverage_pct", "avg_throughput_mbps", "p5_throughput_mbps",
               "handover_count", "peak_load"]
    print("\n" + "=" * 90)
    print(f"{'Metric':<24}" + "".join(f"w={w:<14}" for w in weights))
    for m in metrics:
        row = f"{m:<24}"
        for w in weights:
            vals = [r.get(m, 0.0) for r in out[str(w)]]
            row += f"{np.mean(vals):>8.3f}±{np.std(vals):<6.2f}"
        print(row)
    print("=" * 90)

    with open("wload_sensitivity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
