"""
Module-ablation study (Section IV.F of the paper).

Isolates the incremental contribution of Modules 1, 2, and 3, which are
orthogonal to selector choice and DO compose into one running pipeline
(unlike Modules 2 and 4, which are alternative selectors evaluated head-
to-head in Table IX / Section IV.E, not stacked here).

Configurations:
  A1       Module 1 only: static hierarchical assignment, no reselection
           as UEs move. Throughput is recomputed every tick from the
           UE's current position (physical SNR changes), but the UE
           never re-associates to a different band or base station.
  A1+2     + Module 2: every tick, TOPSIS-scores all bands for the UE
           and applies the A3/A5 hysteresis + time-to-trigger FSM
           (context_aware_handoff.ContextAwareBandSelector, static
           weights) to decide whether to hand off, purely reactively.
  A1+2+3   + Module 3: PredictiveHandoffEngine.tick() also runs every
           tick. When it forecasts an imminent blockage/outage on the
           serving band, that is used ONLY as an early "act now" trigger
           (this is what "Module 3 gates Module 2's handoff timing"
           means concretely) -- Module 2's own TOPSIS ranking (not
           Module 3's best-SNR pick) still decides WHICH band to switch
           to. Absent a Module-3 trigger, behavior is identical to A1+2.

This script does NOT attempt a 4th "Module 4 substituted for Module 2"
row -- that requires a trained DQN policy plugged into the same loop and
is left as a separate, explicitly-flagged follow-up (Table IX already
reports Module 4 vs. heuristics in isolation).

Same K=150 population size, same 20-seed philosophy, and the same KPI
definitions (compute_network_kpis) as Tables V/IX elsewhere in the paper.
"""
import copy
import random
import time as _time
import numpy as np
from scipy import stats

from multiband_planning import (
    HierarchicalMultiBandPlanner, Band, Position, UserEquipment,
    compute_network_kpis, MIN_USABLE_THROUGHPUT_MBPS,
)
from context_aware_handoff import ContextAwareBandSelector, AppType
from predictive_handoff import PredictiveHandoffEngine, Obstacle

# Same 12 seeds already used elsewhere in this project's run_experiments.py,
# extended to 20 (matching the paper's headline 20-seed statistical
# comparisons) with 8 additional arbitrary, fixed, disclosed seeds.
SEEDS = [42, 7, 19, 3, 101, 17, 23, 58, 91, 4, 77, 12,
         55, 88, 33, 66, 99, 111, 222, 5]

N_UES = 150
N_TICKS = 150          # 100 ms/tick -> 5.0 s of simulated mobility per seed
TICK_S = 0.1
AREA_M = 500.0

# Same three obstacles used in main_simulation.py's demo, so Module 3 has
# real geometry to detect against.
OBSTACLES = [
    Obstacle(x_min=150, x_max=200, y_min=150, y_max=300, attenuation_db=30),
    Obstacle(x_min=300, x_max=350, y_min=50,  y_max=150, attenuation_db=25),
    Obstacle(x_min=50,  x_max=100, y_min=350, y_max=450, attenuation_db=20),
]


def _build_population(seed):
    random.seed(seed)
    np.random.seed(seed)
    planner = HierarchicalMultiBandPlanner()
    planner.deploy_grid(area_m=AREA_M, sub6_count=3, mmwave_count=8, thz_count=15)
    ues = []
    for i in range(N_UES):
        pos = Position(x=random.uniform(0, AREA_M), y=random.uniform(0, AREA_M))
        roll = random.random()
        if roll < 0.30:
            speed = random.uniform(0, 4)
        elif roll < 0.70:
            speed = random.uniform(4, 15)
        else:
            speed = random.uniform(15, 30)
        ues.append(UserEquipment(ue_id=i, position=pos, velocity=speed))
    planner.ues = ues
    planner._bs_index = {b.bs_id: b for b in planner.base_stations}
    planner.run_planning()
    app_types = [random.choice(list(AppType)) for _ in ues]
    return planner, ues, app_types


def _move(ue, rng):
    angle = rng.uniform(0, 2 * np.pi)
    ue.position.x = max(0.0, min(AREA_M, ue.position.x + ue.velocity * np.cos(angle) * TICK_S))
    ue.position.y = max(0.0, min(AREA_M, ue.position.y + ue.velocity * np.sin(angle) * TICK_S))


def _recompute_throughput_frozen(planner, ue):
    """A1: BS/band assignment is frozen; only recompute the physical
    throughput the UE currently gets from that same base station."""
    if ue.assigned_bs is None:
        ue.throughput_mbps = 0.0
        return
    bs = planner._bs_index.get(ue.assigned_bs)
    if bs is None:
        ue.throughput_mbps = 0.0
        return
    cap = planner.prop_model.shannon_capacity_mbps(bs, ue)
    n = max(len(bs.active_ues), 1)
    ue.throughput_mbps = cap / n


def _reassign(planner, ue, band):
    """Disconnect from current BS (if any) and connect to the best BS in
    `band`, using Module 1's own best-BS-in-band selection logic so BS
    choice within a band is consistent with the rest of the paper."""
    if ue.assigned_bs is not None and ue.assigned_bs in planner._bs_index:
        old_bs = planner._bs_index[ue.assigned_bs]
        if ue.ue_id in old_bs.active_ues:
            old_bs.active_ues.remove(ue.ue_id)
        ue.assigned_bs, ue.assigned_band, ue.throughput_mbps = None, None, 0.0

    candidates = [bs for bs in planner.base_stations if bs.band == band]
    if not candidates:
        return False
    best_bs = max(candidates, key=lambda b: planner.prop_model.snr_db(b, ue))
    best_bs.active_ues.append(ue.ue_id)
    ue.assigned_bs = best_bs.bs_id
    ue.assigned_band = band
    n = max(len(best_bs.active_ues), 1)
    ue.throughput_mbps = planner.prop_model.shannon_capacity_mbps(best_bs, ue) / n
    return True


def _snapshot_kpis(planner, ues):
    return compute_network_kpis(ues, planner.base_stations)


def _avg_kpis(kpi_list):
    keys = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
            "p5_throughput_mbps", "load_std", "capacity_violation_pct"]
    return {k: float(np.mean([kp[k] for kp in kpi_list])) for k in keys}


def run_config_A1(seed):
    planner, ues, _ = _build_population(seed)
    rng = np.random.default_rng(seed + 10_000)
    snaps = []
    for tick in range(N_TICKS):
        for ue in ues:
            _move(ue, rng)
            _recompute_throughput_frozen(planner, ue)
        snaps.append(_snapshot_kpis(planner, ues))
    out = _avg_kpis(snaps)
    out["handover_count"] = 0
    return out


def run_config_A1_2(seed):
    planner, ues, app_types = _build_population(seed)
    selector = ContextAwareBandSelector(weight_mode="static")
    rng = np.random.default_rng(seed + 20_000)
    snaps, ho_count = [], 0
    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app in zip(ues, app_types):
            _move(ue, rng)
            ctx = selector.build_context(ue, app, planner.base_stations)
            band, ho_event = selector.select_band(ctx, time_ms=t_ms)
            if band is not None and (ue.assigned_band != band or ue.assigned_bs is None):
                if _reassign(planner, ue, band):
                    if ho_event is not None:
                        ho_count += 1
            else:
                _recompute_throughput_frozen(planner, ue)
        snaps.append(_snapshot_kpis(planner, ues))
    out = _avg_kpis(snaps)
    out["handover_count"] = ho_count
    return out


def run_config_A1_2_3(seed):
    planner, ues, app_types = _build_population(seed)
    selector = ContextAwareBandSelector(weight_mode="static")
    engine = PredictiveHandoffEngine(planner.base_stations)
    for ue in ues:
        engine.register_ue(ue)
    for obs in OBSTACLES:
        engine.add_obstacle(obs)

    rng = np.random.default_rng(seed + 30_000)
    snaps, ho_count, proactive_count = [], 0, 0
    for tick in range(N_TICKS):
        t_ms = tick * 100.0
        for ue, app in zip(ues, app_types):
            _move(ue, rng)

            # Module 3: is a proactive handoff warranted right now?
            proactive_decision = engine.tick(ue, time_ms=t_ms)

            if proactive_decision is not None:
                # Module 3 decides WHEN; Module 2's TOPSIS ranking (not
                # Module 3's own best-SNR pick) decides WHICH band.
                ctx = selector.build_context(ue, app, planner.base_stations)
                candidates = [b for b in Band if b in ctx.snr_db]
                scores = selector.scorer.score(ctx, candidates, selector.prop, ctx.serving_band)
                target = next(iter(scores), None) if scores else None
                if target is not None and target != ue.assigned_band:
                    if _reassign(planner, ue, target):
                        ho_count += 1
                        proactive_count += 1
                        continue
                _recompute_throughput_frozen(planner, ue)
                continue

            # Otherwise: same reactive Module 2 behavior as A1+2.
            ctx = selector.build_context(ue, app, planner.base_stations)
            band, ho_event = selector.select_band(ctx, time_ms=t_ms)
            if band is not None and (ue.assigned_band != band or ue.assigned_bs is None):
                if _reassign(planner, ue, band):
                    if ho_event is not None:
                        ho_count += 1
            else:
                _recompute_throughput_frozen(planner, ue)
        snaps.append(_snapshot_kpis(planner, ues))
    out = _avg_kpis(snaps)
    out["handover_count"] = ho_count
    out["proactive_handover_count"] = proactive_count
    return out


def paired_ttest(a, b):
    a, b = np.array(a), np.array(b)
    if np.allclose(a, b):
        return float("nan"), float("nan"), float("nan")
    t, p = stats.ttest_rel(a, b)
    d = np.mean(a - b) / np.std(a - b, ddof=1) if np.std(a - b, ddof=1) > 0 else float("nan")
    return float(t), float(p), float(d)


def main():
    configs = [("A1", run_config_A1), ("A1+2", run_config_A1_2), ("A1+2+3", run_config_A1_2_3)]
    results = {name: [] for name, _ in configs}

    t0 = _time.time()
    for seed in SEEDS:
        for name, fn in configs:
            results[name].append(fn(seed))
        print(f"seed {seed} done ({_time.time()-t0:.1f}s elapsed)")

    metrics = ["coverage_pct", "qos_coverage_pct", "avg_throughput_mbps",
               "p5_throughput_mbps", "load_std", "capacity_violation_pct",
               "handover_count"]

    print("\n" + "=" * 90)
    print(f"{'Metric':<26}" + "".join(f"{name:>18}" for name, _ in configs))
    for m in metrics:
        row = f"{m:<26}"
        for name, _ in configs:
            vals = [r.get(m, 0.0) for r in results[name]]
            row += f"{np.mean(vals):>12.3f} ±{np.std(vals):<4.2f}"
        print(row)
    print("=" * 90)

    print("\nPairwise paired t-tests (adjacent configs):")
    pairs = [("A1", "A1+2"), ("A1+2", "A1+2+3")]
    for m in metrics:
        for c1, c2 in pairs:
            a = [r.get(m, 0.0) for r in results[c1]]
            b = [r.get(m, 0.0) for r in results[c2]]
            t, p, d = paired_ttest(a, b)
            print(f"  {m:<24} {c1:>8} vs {c2:<8}: t={t:.3f}  p={p:.4g}  d={d:.3f}")

    import json
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved raw per-seed results to ablation_results.json")


if __name__ == "__main__":
    main()
