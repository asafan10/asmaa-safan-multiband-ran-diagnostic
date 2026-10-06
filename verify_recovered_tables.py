"""Compares every freshly generated result file in the working directory against the golden per-seed
snapshots in results_snapshots/ (the numbers behind the paper's tables). Exit code 1 on any mismatch."""
import json, math, os, sys

PAIRS = {
    "ablation_results.json": "Table VII rows 1-3 (A1, A1+2, A1+2+3)",
    "ablation_row4_torch_v2_results.json": "Table VII row 4 (Module 4 in closed loop)",
    "ablation_row4b_torch_v2_results.json": "Section V.G retrain at K=150",
    "ablation_row4_cooldown_v2_results.json": "Table A4 (cooldown-gated Module 4)",
    "ablation_row5_results.json": "Table VIII (score-trigger)",
    "ablation_row6_results.json": "Table IX variant 1",
    "ablation_row7_results.json": "Table IX variant 2",
    "ablation_row8_results.json": "Table IX variant 3",
    "table_x_full_stats.json": "Table IX full statistics",
    "wload_sensitivity_results.json": "Table A2 (w_load sweep)",
    "table_vi_v2_all_seeds.json": "Table VI (DQN vs heuristics)",
    "table_vcd_results.json": "Tables V.c / V.d (150-tick protocol)",
    "decomposition_check_results.json": "Gap decomposition (Section V-B)",
    "load_distribution_results.json": "Per-cell load analysis (Fig. 3, Section V-B)",
}

def walk(a, b, path, bad, cnt, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in b:
            if k not in a: bad.append((path + "/" + str(k), "missing", None)); continue
            walk(a[k], b[k], path + "/" + str(k), bad, cnt, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): bad.append((path, "len", (len(a), len(b)))); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", bad, cnt, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        cnt[0] += 1
        if math.isnan(a) and math.isnan(b): return
        if abs(a - b) > tol * max(1.0, abs(b)): bad.append((path, a, b))

tol = float(os.environ.get("VERIFY_TOL", "1e-6"))
fail = 0
for fn, what in PAIRS.items():
    if not os.path.exists(fn):
        print(f"SKIP  {fn:45s} ({what}): not regenerated in this run"); continue
    snap = os.path.join("results_snapshots", fn)
    if not os.path.exists(snap):
        print(f"SKIP  {fn:45s} no golden snapshot"); continue
    a, b = json.load(open(fn)), json.load(open(snap))
    bad, cnt = [], [0]
    walk(a, b, "", bad, cnt, tol)
    status = "MATCH" if not bad else "DIFF "
    print(f"{status} {fn:45s} {cnt[0]:6d} numbers compared, {len(bad)} differ  ({what})")
    for p, x, y in bad[:5]: print("        ", p, x, y)
    fail += bool(bad)
sys.exit(1 if fail else 0)
