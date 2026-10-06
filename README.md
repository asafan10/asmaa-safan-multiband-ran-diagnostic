# Reproducibility package

Accompanies "Destination-Load Awareness and Decoupled Hysteresis: A Diagnostic Study of Cross-Tier Band Selection in Multi-Band 5G/6G RANs".
This README replaces an earlier one that described a NumPy-only environment and an older table numbering; it now matches the manuscript.

## 1. Environment

| Component | Version |
|---|---|
| Python | 3.11 |
| numpy / scipy | 2.4.4 / 1.17.1 (`requirements.txt`) |
| PyTorch | 2.14.0 (`requirements-torch.txt`), CPU only |

Every DQN result in the paper (Table VI, Table VII row 4, Tables A4 and A5) was produced with the **PyTorch** backend;
the closed-loop ablation rows without a DQN (Tables VII rows 1-3, VIII, IX) are NumPy-only. `dqn_traffic_steering.py` falls back to a NumPy MLP when PyTorch is absent; that fallback is used only for the wall-clock timing
comparison of Section V-F. Do not mix the two backends when comparing numbers.

## 2. One-command reproduction

```
./reproduce.sh          # Tables V.a/V.b (NumPy only, seconds) and V.c/V.d (a few minutes)
./reproduce.sh --full   # additionally: Tables VI-IX, A2, A4, A5 and Section V-B/V-D supporting results; ends with
                        # verify_recovered_tables.py, which compares every fresh per-seed value to results_snapshots/
```

## 3. Script-to-table manifest

| Paper item | Script | Notes |
|---|---|---|
| Tables V.a, V.b (one-shot snapshot) | `reproduce_table_v.py` (calls `run_experiments.compare_planners`) | exact |
| Fig. 3 (per-cell load CDF), per-cell load analysis in Section V-B | `load_distribution_analysis.py` | writes `load_distribution_results.json` |
| Tables V.c, V.d (150-tick protocol) | `reproduce_table_vcd.py` | Module 1 row equals Table VII row A1 by construction |
| Gap decomposition (890 vs. 355 Mbps) | `decomposition_check.py` | |
| Fig. 4, Fig. 5 (TOPSIS example, rank stability) | `run_experiments.topsis_sensitivity` | |
| Table VI (DQN vs. heuristics, 3 training seeds) | `reproduce_table_vi.py` | depends on the PyTorch thread count; run at the default |
| Table VII rows 1-3 | `ablation_experiment.py` | |
| Table VII row 4 | `ablation_row4_torch_v2.py` | single thread pinned inside the script |
| Table A5 (retraining at K = 150) | `ablation_row4b_torch_v2.py` | |
| Table A4 (cooldown) | `ablation_row4_cooldown_v2.py` | |
| Table VIII | `ablation_row5_scoretrigger.py` | |
| Table IX | `ablation_row6_loadaware.py`, `ablation_row7_cellload.py`, `ablation_row8_power2.py`, `table_x_full_stats.py` | `table_x_full_stats.py` is the aggregation script |
| Table A2 (w_load sweep) | `wload_sensitivity.py` | first 8 seeds |
| Section V-D (Kalman vs. exponential smoothing, Fig. 6), blockage prediction | `run_experiments.compare_position_trackers`, `run_experiments.evaluate_blockage_prediction(guarantee_blockage_crossings=True)` | |
| Section V-F (timing) | `run_experiments.benchmark_timing` | NumPy MLP backend |
| Table A3 (pooled Holm-Bonferroni) | arithmetic on the p-values of the tables above (Eq. (A14)) | no separate script |
| Tables A1, A6.a, A6.b, A7, A8 | no script in this package (see VERIFICATION_REPORT.md) | |

## 4. Verification status (this revision)

See `VERIFICATION_REPORT.md`. Items without an automated script in this package are listed there explicitly rather than implied to be covered. Table VI uses 20 evaluation episodes (seeds 9000-9019).

## 5. Seeds

The canonical 20-seed list is `[42, 7, 19, 3, 101, 17, 23, 58, 91, 4, 77, 12, 55, 88, 33, 66, 99, 111, 222, 5]`, defined in
`ablation_experiment.SEEDS` and `run_experiments.SEEDS`. Table VI uses training seeds 42, 7, 19; evaluation seeds 9000-9019.
Per-script seeding details are in the script headers and in `SEED_MANIFEST.md`.

## 6. Bit-reproducibility

Within one machine and thread count, results are bit-reproducible. Across CPUs, BLAS builds or PyTorch thread counts they are only statistically
reproducible. Table VII row 4 is therefore pinned to one thread; Table VI deliberately is not (see the manuscript's Data and Code Availability).

## 7. Dockerfile

`Dockerfile.draft` has not been built or tested. `reproduce.sh` against the pinned requirements is the supported route.

## 8. What this package reproduces and what it does not

- It reproduces the tables and statistics listed in the manifest of Section 3, including all Section V statistics (`reproduce.sh`; `reproduce.sh --full` also regenerates the DQN-based and recovered-script results and verifies them against per-seed reference values). Tables A1, A6.a, A6.b, A7 and A8 have no script here and are listed in `VERIFICATION_REPORT.md`.
- It includes the four evaluation-artifact checks of Section V-J.
- It does **not** include a DQN retrained with a rescaled reward. The reported DQN result is for a policy trained at K = 10 and transferred unchanged to K = 150, plus the K = 150 retraining of Table A5. The reward-scale confound (C_ref = 10,000 Mbps) is described in the paper.
- It does **not** include a comparison against a published state-of-the-art scheme. All comparisons are against internal baselines, as stated in limitation 8 of the paper.

