"""
One-command reproduction of Table V.a/b (Module 1 vs. BL-1 Max-SNR and
BL-4 Load-Aware SNR, K=150, 20 seeds).

Chosen over Table IX for `reproduce.sh` because it runs in well under a
second on ordinary hardware (no DQN training involved), so a reader gets
immediate end-to-end confirmation that the environment is set up
correctly before attempting anything training-based.
"""
import sys
from run_experiments import compare_planners, SEEDS, COMPARISON_N_UES

if __name__ == "__main__":
    print(f"Reproducing Table V.a/b: {len(SEEDS)} seeds, n_ues={COMPARISON_N_UES}")
    if len(SEEDS) != 20:
        sys.exit(f"FAIL: expected 20 seeds, found {len(SEEDS)} -- SEEDS list is stale.")
    results = compare_planners()
    print("\nDone. Compare the 'Proposed' column means above against Table V.a/b in the paper.")
