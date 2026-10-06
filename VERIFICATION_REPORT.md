# Verification report (re-run for the revised manuscript)

Environment: 2-core CPU container, Python 3.11, numpy 2.4.4, scipy 1.17.1, torch 2.14.0 (CPU). `./reproduce.sh --full` took 82 minutes.
"Golden" means the per-seed JSON files in `results_snapshots/`, compared number-by-number by `verify_recovered_tables.py`
(relative tolerance 1e-9 unless stated in that script).

| Paper item | Script | Result of this re-run |
|---|---|---|
| Tables V.a, V.b | `reproduce_table_v.py` | **Identical** to the printed table values (all means, SDs, t, p, Wilcoxon p, d). |
| Tables V.c, V.d | `reproduce_table_vcd.py` (new) | **Identical** to the revised tables and to the golden JSON (380 numbers). The old BL-1/BL-4 rows could not be regenerated (script lost) and were replaced. |
| 890 → 355 Mbps decomposition | `decomposition_check.py` (new) | 355.05 / 474.28 / 355.85 Mbps; matches the text; golden JSON identical (60 numbers). |
| Per-cell load analysis, Fig. 3 | `load_distribution_analysis.py` | 85.4 % vs 48.7 % zero-load cells, peak 0.43 vs 0.80, 1 of 520 cells ≥ 0.8; golden identical (1095 numbers). |
| Table VI | `reproduce_table_vi.py` | **Identical** to the paper for all three training seeds (DQN 1857.43 / 1645.46 / 1767.84 Mbps; heuristics, t, p, d for seed 42). *Found and fixed:* `compare_dqn_vs_heuristics` defaulted to 12 evaluation episodes in this package version; the paper uses 20 (seeds 9000–9019). Default is now 20, snapshot regenerated. Thread-count dependence (README §6) still applies. |
| Table VII rows 1–3 | `ablation_experiment.py` | Match golden (440 numbers). |
| Table VII row 4 | `ablation_row4_torch_v2.py` | Match golden (160 numbers; single thread). |
| Table A5 (K = 150 retraining) | `ablation_row4b_torch_v2.py` | Match golden (160 numbers). |
| Table A4 (cooldown) | `ablation_row4_cooldown_v2.py` | Match golden (480 numbers). |
| Table VIII | `ablation_row5_scoretrigger.py` | Match golden (301 numbers). Paper value 83.22 corrected to 83.21 (true mean 83.2149). |
| Table IX | `ablation_row6/7/8_*.py`, `table_x_full_stats.py` | `table_x_full_stats.py` match golden (720 numbers); the three single-variant result files are regenerated inside it and not compared separately. |
| Table A2 (w_load sweep) | `wload_sensitivity.py` | Match golden (216 numbers). |
| Fig. 5 / Section V-B rank stability | `reproduce_misc.py` | 0.966 mean, 0.502 minimum, CV mean 3.75 % (p95 5.92 %) as in the text. |
| Section V-D Kalman vs. exponential smoothing | `reproduce_misc.py` | −8.7 %, p = 0.018, **d = −0.58** (paper had −0.59; corrected). |
| Blockage prediction | `reproduce_misc.py` | 2 true positives of 21 genuine events, 17 firings, as in the text (requires `guarantee_blockage_crossings=True`). |
| Section V-F timing | `reproduce_misc.py` | Same order of magnitude, not identical (hardware-dependent): planning 0.70 ms for 20 UEs, inference 0.0055 ms, training step 2.35 ms (ratio about 430×). The paper text was changed to “2–3 ms, about 300–430×”. |
| Table A3 (pooled Holm) | arithmetic | Recomputed from per-seed snapshots and the printed p-values of Tables V.a/V.b/VI: **26 of 42** significant pooled (28 per table), not 25. Paper corrected. |

## Not covered by any script in this package

Table A1 (band distribution by velocity), Tables A6.a and A6.b (sensitivity sweeps), Tables A7 and A8 (within-tick-cache comparisons), Fig. 2 (K = 50 illustration), Fig. 4 and Fig. 6 plots, the 8–11 dB / 33–79 dB link-budget statements, and the per-seed ranges quoted for Table V.c/V.d beyond what `reproduce_table_vcd.py` prints. They are as reported in the manuscript and were checked only for internal consistency (text against tables).
