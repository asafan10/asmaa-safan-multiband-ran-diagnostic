"""
Ablation row 5 (supplementary): does replacing Module 2's A3 SNR-magnitude
gate with a pure TOPSIS-score-improvement trigger fix the near-zero
handover-activation finding of Section IV.G (Table XIII), without
otherwise changing anything else in the pipeline?

Answer, tested directly rather than assumed: it fixes the ACTIVATION-RATE
symptom (handovers per seed jump from ~0.3-0.5 to several hundred) but at
the cost of a NEW, more severe failure: without any real-time destination-
load term in TOPSIS's own criteria vector (throughput, latency, SNR,
battery, handover-cost -- no load/congestion term), many UEs that share
similar propagation geometry independently identify the SAME locally-best
base station as their top-ranked target in the same tick window and pile
onto it. This is measured directly (not inferred): with the score-margin
trigger active, individual mmWave base stations are observed with 20-40+
simultaneously active UEs, i.e., 4-8x this deployment's typical per-cell
load, which collapses per-UE throughput under the paper's equal-time-
sharing capacity model.

This is implemented as a monkeypatch of HandoffStateMachine.evaluate
(not a permanent change to context_aware_handoff.py, since that module is
shared by every other table/figure in the paper and this variant is not
adopted) so it can be run against the exact same ablation harness
(ablation_experiment.py) with no other code path affected.
"""
import time as _time
import numpy as np

import context_aware_handoff as _cah
from context_aware_handoff import HandoffStateMachine, HandoffEvent, HandoffState, HandoffTrigger

from ablation_experiment import SEEDS, N_TICKS, run_config_A1_2, run_config_A1_2_3, paired_ttest

SCORE_MARGIN = 0.05


def _patched_evaluate(self, ctx, scores, current_time_ms=0.0):
    if not ctx.serving_band or not scores:
        return None
    best_band = next(iter(scores))
    best_score = scores[best_band]
    curr_score = scores.get(ctx.serving_band, 0.0)
    if best_band == ctx.serving_band:
        self.state = HandoffState.IDLE
        self.pending_trigger = None
        return None

    # Score-only A3: no SNR-magnitude requirement at all (this is the
    # variant under test).
    a3_fired = best_score > curr_score + SCORE_MARGIN
    a5_fired = (ctx.snr_db.get(ctx.serving_band, 0) < self.A5_THRESHOLD1_DB and
                ctx.snr_db.get(best_band, -99) > self.A5_THRESHOLD2_DB)

    if a3_fired or a5_fired:
        event = HandoffEvent.A3 if a3_fired else HandoffEvent.A5
        if self.state == HandoffState.IDLE:
            ttt = self.TTT_MS.get((ctx.serving_band, best_band), 40)
            self.pending_trigger = HandoffTrigger(
                event=event, source_band=ctx.serving_band, target_band=best_band,
                trigger_time=current_time_ms, ttt_ms=ttt,
            )
            self.trigger_start = current_time_ms
            self.state = HandoffState.TRIGGERED
        elif self.state == HandoffState.TRIGGERED:
            elapsed = current_time_ms - (self.trigger_start or 0)
            ttt_needed = self.pending_trigger.ttt_ms if self.pending_trigger else 40
            if elapsed >= ttt_needed:
                src, tgt = self.pending_trigger.source_band, self.pending_trigger.target_band
                self.state = HandoffState.IDLE
                self.pending_trigger = None
                return (src, tgt)
    else:
        if self.state == HandoffState.TRIGGERED:
            self.state = HandoffState.IDLE
            self.pending_trigger = None
    return None


def main():
    HandoffStateMachine.evaluate = _patched_evaluate  # monkeypatch, this process only

    configs = [("A1+2 (score-trigger)", run_config_A1_2),
               ("A1+2+3 (score-trigger)", run_config_A1_2_3)]
    results = {name: [] for name, _ in configs}

    # Also record peak per-BS load directly, on one representative seed,
    # as concrete evidence for the "congestion stampede" mechanism.
    import ablation_experiment as ae
    planner, ues, app_types = ae._build_population(SEEDS[0])
    selector = _cah.ContextAwareBandSelector(weight_mode="static")
    rng = np.random.default_rng(SEEDS[0] + 20_000)
    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app in zip(ues, app_types):
            ae._move(ue, rng)
            ctx = selector.build_context(ue, app, planner.base_stations)
            band, ho_event = selector.select_band(ctx, time_ms=t_ms)
            if band is not None and (ue.assigned_band != band or ue.assigned_bs is None):
                ae._reassign(planner, ue, band)
            else:
                ae._recompute_throughput_frozen(planner, ue)
    peak_load = max(len(bs.active_ues) for bs in planner.base_stations)
    print(f"Peak single-BS active-UE count on seed {SEEDS[0]} (score-trigger variant): {peak_load}")

    t0 = _time.time()
    for seed in SEEDS:
        for name, fn in configs:
            results[name].append(fn(seed))
        print(f"seed {seed} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct", "handover_count"]

    print("\n" + "=" * 90)
    print(f"{'Metric':<26}" + "".join(f"{name:>24}" for name, _ in configs))
    for m in metrics:
        row = f"{m:<26}"
        for name, _ in configs:
            vals = [r.get(m, 0.0) for r in results[name]]
            row += f"{np.mean(vals):>16.3f} ±{np.std(vals):<6.2f}"
        print(row)
    print("=" * 90)

    import json
    with open("ablation_row5_results.json", "w") as f:
        json.dump({"peak_load": peak_load, "results": results}, f, indent=2)

    # Compare against the original (unmitigated, SNR-gated) A1+2 / A1+2+3
    # rows already saved from ablation_experiment.py's run.
    try:
        with open("ablation_results.json") as f:
            original = json.load(f)
        print("\nPaired t-test vs. original SNR-gated A1+2 / A1+2+3 (same 20 seeds):")
        pairs = [("A1+2", "A1+2 (score-trigger)"), ("A1+2+3", "A1+2+3 (score-trigger)")]
        for orig_name, new_name in pairs:
            print(f"  -- {orig_name} vs {new_name} --")
            for m in metrics:
                a = [r.get(m, 0.0) for r in original[orig_name]]
                b = [r.get(m, 0.0) for r in results[new_name]]
                t, p, d = paired_ttest(a, b)
                print(f"    {m:<24} t={t:.3f}  p={p:.4g}  d={d:.3f}")
    except FileNotFoundError:
        print("\n(ablation_results.json not found -- skip comparison)")


if __name__ == "__main__":
    main()
