"""
Tables V.c / V.d: Module 1 vs. BL-1 (Max-SNR) and BL-3 (Load-Aware SNR) under the 150-tick mobility protocol.

Protocol (identical for the three methods):
  * K=150 population and topology of the closed-loop ablation (ablation_experiment._build_population, 20 seeds).
  * Each method performs its own one-shot association at t=0; assignments are then frozen for 150 ticks (100 ms).
  * UE motion is the same random-heading walk (same movement RNG, seed+10000); shadow fading is redrawn
    independently at every throughput read.
  * Module 1's row IS ablation_experiment's A1 row (Table VII), by construction.
  * To make each baseline independent of call order, the global RNGs are re-seeded with the seed value
    immediately before each baseline's association step (see paper, Section V.J, second mechanism).
Statistics: paired t-test, Wilcoxon signed-rank (two-sided), Cohen's d = mean(diff)/SD(diff, ddof=1),
Bonferroni alpha' = 0.05/6 (3 testable metrics x 2 baselines). Population SD is printed for mean +/- SD.
"""
import json, random
import numpy as np
from scipy import stats
import ablation_experiment as ab
from multiband_planning import MaxSNRPlanner, LoadAwareSNRPlanner, clone_topology

METRICS = [("qos_coverage_pct", "Coverage, >=25 Mbps (%)"), ("avg_throughput_mbps", "Avg. throughput (Mbps)"),
           ("p5_throughput_mbps", "P5 throughput (Mbps)")]

def loop(planner, ues, seed):
    rng = np.random.default_rng(seed + 10_000)
    snaps = []
    for _ in range(ab.N_TICKS):
        for ue in ues:
            ab._move(ue, rng)
            ab._recompute_throughput_frozen(planner, ue)
        snaps.append(ab._snapshot_kpis(planner, ues))
    return ab._avg_kpis(snaps)

def baseline(seed, cls):
    planner, ues, _ = ab._build_population(seed)
    bss, cu = clone_topology(planner.base_stations, ues)
    random.seed(seed); np.random.seed(seed)
    b = cls(bss); b.ues = cu
    b.run_planning()
    return loop(b, cu, seed)

def main():
    res = {"A1": [ab.run_config_A1(s) for s in ab.SEEDS],
           "BL-1": [baseline(s, MaxSNRPlanner) for s in ab.SEEDS],
           "BL-3": [baseline(s, LoadAwareSNRPlanner) for s in ab.SEEDS]}
    json.dump(res, open("table_vcd_results.json", "w"), indent=1)
    a = 0.05 / 6
    for name in ("BL-1", "BL-3"):
        print(f"\nModule 1 (A1) vs {name}")
        for k, lab in METRICS:
            x = np.array([r[k] for r in res["A1"]]); y = np.array([r[k] for r in res[name]]); d = x - y
            t, p = stats.ttest_rel(x, y); w = stats.wilcoxon(x, y).pvalue
            print(f"  {lab:26s} {x.mean():8.2f} ± {x.std():6.2f} | {y.mean():8.2f} ± {y.std():6.2f} | t={t:6.2f} p={p:.2e} "
                  f"W-p={w:.2e} d={d.mean()/d.std(ddof=1):5.2f} sig={'yes' if p < a else 'no'}")
        q = np.array([r["qos_coverage_pct"] for r in res["A1"]]) - np.array([r["qos_coverage_pct"] for r in res[name]])
        print(f"  per-seed QoS gap: min {q.min():.1f}  mean {q.mean():.1f}  max {q.max():.1f}")
        print(f"  per-seed QoS coverage ranges: Module 1 {min(r['qos_coverage_pct'] for r in res['A1']):.1f}-"
              f"{max(r['qos_coverage_pct'] for r in res['A1']):.1f}; {name} {min(r['qos_coverage_pct'] for r in res[name]):.1f}-"
              f"{max(r['qos_coverage_pct'] for r in res[name]):.1f}")

if __name__ == "__main__":
    main()
