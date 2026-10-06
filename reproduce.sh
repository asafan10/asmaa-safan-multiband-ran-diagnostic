#!/usr/bin/env bash
# Default path: Tables V.a/V.b (NumPy only, seconds) and V.c/V.d (a few minutes).
# --full: additionally regenerates every DQN-based and recovered-script result (Tables VI, VII, VIII, IX, A2, A4, A5),
#         the Section V-B/V-D supporting results, and verifies each result file against the golden per-seed snapshots
#         in results_snapshots/ (verify_recovered_tables.py). Takes about an hour on 2 CPU cores.
# Usage:  ./reproduce.sh [--full]
set -euo pipefail
cd "$(dirname "$0")"
FULL=0; [[ "${1:-}" == "--full" ]] && FULL=1

python3 --version
python3 -m pip install -r requirements.txt --break-system-packages --quiet
[[ "$FULL" -eq 1 ]] && python3 -m pip install -r requirements-torch.txt --break-system-packages --quiet

echo "== Tables V.a/V.b (20 seeds, K=150) =="; python3 reproduce_table_v.py
echo "== Tables V.c/V.d (150-tick protocol) =="; python3 reproduce_table_vcd.py
[[ "$FULL" -eq 0 ]] && { echo "== Done (fast path). Use --full for the DQN-based tables. =="; exit 0; }

echo "== Gap decomposition (Section V-B) =="; python3 decomposition_check.py
echo "== Per-cell load analysis (Fig. 3) =="; python3 load_distribution_analysis.py
echo "== Table VII rows 1-3 (A1, A1+2, A1+2+3) =="; python3 ablation_experiment.py
echo "== Table VI (real PyTorch DQN vs heuristics; default thread count, see README) =="; python3 reproduce_table_vi.py
echo "== Table VII row 4 (single thread) =="; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ablation_row4_torch_v2.py
echo "== Table A5 (K=150 retraining) =="; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ablation_row4b_torch_v2.py
echo "== Table A4 (cooldown-gated Module 4) =="; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 ablation_row4_cooldown_v2.py
echo "== Table VIII (score-trigger) =="; python3 ablation_row5_scoretrigger.py
echo "== Table IX (destination-load variants) =="; python3 table_x_full_stats.py
echo "== Table A2 (w_load sweep) =="; python3 wload_sensitivity.py
echo "== Section V-B/V-D/V-F supporting results =="; python3 reproduce_misc.py
echo "== Verification against golden snapshots =="; python3 verify_recovered_tables.py
