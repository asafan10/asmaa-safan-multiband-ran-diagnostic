"""
Ablation row 6 (follow-up to row 5 / Section IV.H): does adding a
real-time destination-load criterion to TOPSIS fix the congestion
stampede that the pure score-improvement trigger (row 5) introduced?

Section IV.J's own generalizable finding is that a cross-tier
band-selection criterion needs BOTH ingredients at once:
  (i)  a real-time destination-load / congestion-risk term, and
  (ii) a hysteresis mechanism decoupled from raw signal-strength
       magnitude.
Row 5 tested (ii) alone (score-only A3 trigger) and found it fails
without (i): UEs pile onto the same locally-best cell. This script
tests (i) + (ii) together: the same score-only trigger from row 5,
PLUS a 6th TOPSIS criterion for destination-cell load, to see whether
the combination actually fixes the congestion stampede rather than
merely being asserted to.

Both changes are implemented as monkeypatches (not permanent edits to
context_aware_handoff.py, which is shared by every other table/figure
in the paper and is left exactly as originally validated), so this can
be run against the identical ablation harness (ablation_experiment.py)
with no other code path affected -- same pattern as row 5.

Load-aware TOPSIS criterion, concretely:
  - For each candidate band, the destination cell is the same
    max-SNR base station in that band that _reassign() would actually
    connect the UE to (so the criterion reflects the cell the UE would
    really land on, not some other cell in the same band).
  - That cell's existing `BaseStation.load` property (fractional
    load already used by Module 1's own load-aware ranking, Eq. 16)
    is used directly as a 6th, cost-type (lower-is-better) criterion.
  - Weight vector is extended from 5 to 6 entries and renormalized;
    load is given weight comparable to SNR/latency (not dominant),
    since the point is to break ties among otherwise-similar bands
    rather than override throughput/latency/SNR outright.
"""
import time as _time
import numpy as np

import context_aware_handoff as _cah
from context_aware_handoff import (
    HandoffStateMachine, HandoffEvent, HandoffState, HandoffTrigger,
    TOPSISBandScorer, Band,
)
from ablation_experiment import SEEDS, N_TICKS, run_config_A1_2, run_config_A1_2_3, paired_ttest

SCORE_MARGIN = 0.05

# Extended weight vector [throughput, latency, SNR, battery, HO cost, load].
# Renormalized to sum to 1.0; load given weight 0.15, taken proportionally
# from the other five so their relative ratios are preserved.
_BASE = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
LOAD_WEIGHT = 0.15
EXTENDED_WEIGHTS = np.concatenate([_BASE * (1 - LOAD_WEIGHT), [LOAD_WEIGHT]])
assert abs(EXTENDED_WEIGHTS.sum() - 1.0) < 1e-9


def _patched_build_context(self, ue, app_type, bss):
    """Same as the original, plus a per-band destination-cell-load
    lookup attached to the context (ctx.band_load), using the same
    max-SNR-in-band cell selection _reassign() uses."""
    ctx = _cah.ContextAwareBandSelector.build_context.__wrapped_orig__(self, ue, app_type, bss)
    ctx.band_load = {}
    for band in Band:
        band_bss = [b for b in bss if b.band == band]
        if not band_bss:
            continue
        best_bs = max(band_bss, key=lambda b: self.prop.snr_db(b, ue))
        ctx.band_load[band] = best_bs.load
    return ctx


def _patched_score(self, ctx, candidate_bands, prop_model, serving_band):
    """Same TOPSIS procedure as the original 5-criterion scorer, with a
    6th column appended for destination-cell load (cost criterion)."""
    if not candidate_bands:
        return {}
    qos = _cah.QOS_PROFILES[ctx.app_type]
    band_load = getattr(ctx, "band_load", {})

    matrix = []
    for band in candidate_bands:
        snr = ctx.snr_db.get(band, -20)
        tput = self._estimate_throughput(band, snr)
        lat = _cah.LATENCY_MS[band]
        bat = _cah.POWER_WEIGHT[band] * (1.0 - ctx.battery_pct / 100.0)
        ho = _cah.HO_PENALTY_MS.get((serving_band, band), 0) if serving_band and serving_band != band else 0
        load = band_load.get(band, 0.0)
        matrix.append([tput, lat, snr, bat, ho, load])

    matrix = np.array(matrix, dtype=float)
    n_bands, n_crit = matrix.shape

    col_norm = np.linalg.norm(matrix, axis=0)
    col_norm[col_norm == 0] = 1
    norm_matrix = matrix / col_norm

    w = EXTENDED_WEIGHTS[:n_crit]
    w = w / w.sum()
    weighted = norm_matrix * w

    benefit_idx = [0, 2]
    cost_idx = [1, 3, 4, 5]
    ideal_best = weighted.max(axis=0).copy()
    ideal_worst = weighted.min(axis=0).copy()
    for idx in cost_idx:
        ideal_best[idx], ideal_worst[idx] = weighted[:, idx].min(), weighted[:, idx].max()

    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)
    denom = d_best + d_worst
    denom[denom == 0] = 1e-9
    scores = d_worst / denom

    qos_scores = {}
    for i, band in enumerate(candidate_bands):
        tput_est = matrix[i, 0]
        lat_est = matrix[i, 1]
        feasible = (tput_est >= qos.min_throughput_mbps * 0.8 and
                    lat_est <= qos.max_latency_ms)
        qos_scores[band] = float(scores[i]) if feasible else 0.0
    return dict(sorted(qos_scores.items(), key=lambda x: -x[1]))


def _patched_evaluate(self, ctx, scores, current_time_ms=0.0):
    """Identical score-only A3 trigger used in ablation row 5."""
    if not ctx.serving_band or not scores:
        return None
    best_band = next(iter(scores))
    best_score = scores[best_band]
    curr_score = scores.get(ctx.serving_band, 0.0)
    if best_band == ctx.serving_band:
        self.state = HandoffState.IDLE
        self.pending_trigger = None
        return None

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
    # Wrap (not replace) build_context so we can call the original then extend it.
    _orig_build_context = _cah.ContextAwareBandSelector.build_context
    _patched_build_context.__wrapped_orig__ = _orig_build_context
    _cah.ContextAwareBandSelector.build_context = _patched_build_context
    TOPSISBandScorer.score = _patched_score
    HandoffStateMachine.evaluate = _patched_evaluate

    configs = [("A1+2 (score-trigger+load)", run_config_A1_2),
               ("A1+2+3 (score-trigger+load)", run_config_A1_2_3)]
    results = {name: [] for name, _ in configs}

    # Peak per-BS load on one representative seed, as direct evidence of
    # whether the congestion stampede is actually resolved (row 5's
    # equivalent check reported 20-40+ simultaneous UEs on one cell).
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
    print(f"Peak single-BS active-UE count on seed {SEEDS[0]} (score-trigger+load-aware variant): {peak_load}")

    t0 = _time.time()
    for seed in SEEDS:
        for name, fn in configs:
            results[name].append(fn(seed))
        print(f"seed {seed} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct", "handover_count"]

    print("\n" + "=" * 90)
    print(f"{'Metric':<26}" + "".join(f"{name:>28}" for name, _ in configs))
    for m in metrics:
        row = f"{m:<26}"
        for name, _ in configs:
            vals = [r.get(m, 0.0) for r in results[name]]
            row += f"{np.mean(vals):>20.3f} ±{np.std(vals):<6.2f}"
        print(row)
    print("=" * 90)

    import json
    with open("ablation_row6_results.json", "w") as f:
        json.dump({"peak_load": peak_load, "results": results,
                    "weights": EXTENDED_WEIGHTS.tolist()}, f, indent=2)

    # Compare against BOTH the original SNR-gated results AND row 5's
    # score-trigger-only (no load term) results, same 20 seeds.
    comparisons = [
        ("original SNR-gated", "ablation_results.json", None,
         [("A1+2", "A1+2 (score-trigger+load)"), ("A1+2+3", "A1+2+3 (score-trigger+load)")]),
        ("score-trigger only (row 5, no load term)", "ablation_row5_results.json", "results",
         [("A1+2 (score-trigger)", "A1+2 (score-trigger+load)"),
          ("A1+2+3 (score-trigger)", "A1+2+3 (score-trigger+load)")]),
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
