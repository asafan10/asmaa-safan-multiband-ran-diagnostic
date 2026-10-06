"""
Consolidated re-run for Table X, addressing reviewer round-5 concerns:
  1. Proper paired t-tests (vs. the score-trigger-only baseline) for every
     metric in Table X, matching the rigor used everywhere else in the paper.
  2. Peak single-cell load reported as mean +/- SD across all 20 seeds
     (not a single-seed number), consistent with every other row.

Baseline = score-trigger-only (row 5, HandoffStateMachine.evaluate patched,
no load criterion). Variants = row 6 (band-level load), row 7 (+ greedy
cell), row 8 (+ power-of-two-choices). All four conditions are re-run here
in one process, in each case tracking BOTH the standard KPI set AND the
peak simultaneous per-BS UE count, per seed, for the A1+2+3 configuration
(the one reported in Table X).
"""
import time as _time
import numpy as np
import json

import context_aware_handoff as _cah
from context_aware_handoff import (
    HandoffStateMachine, HandoffEvent, HandoffState, HandoffTrigger,
    TOPSISBandScorer, Band,
)
import ablation_experiment as ae
from ablation_experiment import SEEDS, N_TICKS, paired_ttest

# ---- Save pristine originals up front ----
_ORIG_build_context = _cah.ContextAwareBandSelector.build_context
_ORIG_score = TOPSISBandScorer.score
_ORIG_evaluate = HandoffStateMachine.evaluate
_ORIG_reassign = ae._reassign

SCORE_MARGIN = 0.05


def _score_only_evaluate(self, ctx, scores, current_time_ms=0.0):
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
                trigger_time=current_time_ms, ttt_ms=ttt)
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


_BASE = np.array([0.35, 0.25, 0.20, 0.10, 0.10])


def make_load_aware_score(load_weight):
    ext = np.concatenate([_BASE * (1 - load_weight), [load_weight]])
    assert abs(ext.sum() - 1.0) < 1e-9

    def _patched_build_context(self, ue, app_type, bss):
        ctx = _ORIG_build_context(self, ue, app_type, bss)
        ctx.band_load = {}
        for band in Band:
            band_bss = [b for b in bss if b.band == band]
            if not band_bss:
                continue
            best_bs = max(band_bss, key=lambda b: self.prop.snr_db(b, ue))
            ctx.band_load[band] = best_bs.load
        return ctx

    def _patched_score(self, ctx, candidate_bands, prop_model, serving_band):
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
        w = ext[:n_crit]
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
            feasible = (tput_est >= qos.min_throughput_mbps * 0.8 and lat_est <= qos.max_latency_ms)
            qos_scores[band] = float(scores[i]) if feasible else 0.0
        return dict(sorted(qos_scores.items(), key=lambda x: -x[1]))

    return _patched_build_context, _patched_score, ext


HYSTERESIS_MARGIN_DB = 3.0


def _reassign_cell_greedy(planner, ue, band):
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


import random as _random


def _reassign_power2(planner, ue, band):
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
        a, b = _random.sample(acceptable, 2)
        best_bs = a if a.load <= b.load else b
    else:
        best_bs = acceptable[0]
    best_bs.active_ues.append(ue.ue_id)
    ue.assigned_bs = best_bs.bs_id
    ue.assigned_band = band
    n = max(len(best_bs.active_ues), 1)
    ue.throughput_mbps = planner.prop_model.shannon_capacity_mbps(best_bs, ue) / n
    return True


def run_A1_2_3_with_peak(seed):
    """Copy of ablation_experiment.run_config_A1_2_3, instrumented to also
    return the peak simultaneous per-BS active-UE count for this seed."""
    from predictive_handoff import PredictiveHandoffEngine
    planner, ues, app_types = ae._build_population(seed)
    selector = _cah.ContextAwareBandSelector(weight_mode="static")
    engine = PredictiveHandoffEngine(planner.base_stations)
    for ue in ues:
        engine.register_ue(ue)
    for obs in ae.OBSTACLES:
        engine.add_obstacle(obs)

    rng = np.random.default_rng(seed + 30_000)
    snaps, ho_count, proactive_count = [], 0, 0
    peak_load = 0
    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app in zip(ues, app_types):
            ae._move(ue, rng)
            proactive_decision = engine.tick(ue, time_ms=t_ms)
            if proactive_decision is not None:
                ctx = selector.build_context(ue, app, planner.base_stations)
                candidates = [b for b in Band if b in ctx.snr_db]
                scores = selector.scorer.score(ctx, candidates, selector.prop, ctx.serving_band)
                target = next(iter(scores), None) if scores else None
                if target is not None and target != ue.assigned_band:
                    if ae._reassign(planner, ue, target):
                        ho_count += 1
                        proactive_count += 1
                        continue
                ae._recompute_throughput_frozen(planner, ue)
                continue
            ctx = selector.build_context(ue, app, planner.base_stations)
            band, ho_event = selector.select_band(ctx, time_ms=t_ms)
            if band is not None and (ue.assigned_band != band or ue.assigned_bs is None):
                if ae._reassign(planner, ue, band):
                    if ho_event is not None:
                        ho_count += 1
            else:
                ae._recompute_throughput_frozen(planner, ue)
        snaps.append(ae._snapshot_kpis(planner, ues))
        cur_peak = max(len(bs.active_ues) for bs in planner.base_stations)
        peak_load = max(peak_load, cur_peak)
    out = ae._avg_kpis(snaps)
    out["handover_count"] = ho_count
    out["proactive_handover_count"] = proactive_count
    out["peak_load"] = peak_load
    return out


def reset_patches():
    _cah.ContextAwareBandSelector.build_context = _ORIG_build_context
    TOPSISBandScorer.score = _ORIG_score
    HandoffStateMachine.evaluate = _ORIG_evaluate
    ae._reassign = _ORIG_reassign


def run_condition(name, apply_patches_fn):
    reset_patches()
    apply_patches_fn()
    t0 = _time.time()
    per_seed = []
    for seed in SEEDS:
        _random.seed(seed)
        per_seed.append(run_A1_2_3_with_peak(seed))
    print(f"[{name}] done in {_time.time()-t0:.1f}s")
    return per_seed


def apply_baseline():
    HandoffStateMachine.evaluate = _score_only_evaluate


def apply_row6():
    bc, sc, ext = make_load_aware_score(0.15)
    _cah.ContextAwareBandSelector.build_context = bc
    TOPSISBandScorer.score = sc
    HandoffStateMachine.evaluate = _score_only_evaluate


def apply_row7():
    apply_row6()
    ae._reassign = _reassign_cell_greedy


def apply_row8():
    apply_row6()
    ae._reassign = _reassign_power2


def main():
    conditions = [
        ("baseline", apply_baseline),
        ("band_load", apply_row6),
        ("greedy_cell", apply_row7),
        ("power2", apply_row8),
    ]
    all_results = {}
    for name, fn in conditions:
        all_results[name] = run_condition(name, fn)

    metrics = ["qos_coverage_pct", "avg_throughput_mbps", "p5_throughput_mbps",
               "handover_count", "peak_load"]

    print("\n" + "=" * 100)
    print(f"{'Metric':<24}" + "".join(f"{name:>19}" for name, _ in conditions))
    for m in metrics:
        row = f"{m:<24}"
        for name, _ in conditions:
            vals = [r.get(m, 0.0) for r in all_results[name]]
            row += f"{np.mean(vals):>13.3f} ±{np.std(vals):<4.2f}"
        print(row)
    print("=" * 100)

    print("\nPaired t-tests vs. baseline (score-trigger-only), same 20 seeds:")
    for name, _ in conditions[1:]:
        print(f"  -- baseline vs {name} --")
        for m in metrics:
            a = [r.get(m, 0.0) for r in all_results["baseline"]]
            b = [r.get(m, 0.0) for r in all_results[name]]
            t, p, d = paired_ttest(a, b)
            print(f"    {m:<22} t={t:.3f}  p={p:.4g}  d={d:.3f}")

    with open("table_x_full_stats.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to table_x_full_stats.json")


if __name__ == "__main__":
    main()
