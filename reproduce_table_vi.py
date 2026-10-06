"""Regenerates Table VI (real PyTorch dueling/double-DQN vs. four heuristics, training seeds 42/7/19).
Writes table_vi_v2_all_seeds.json in the working directory. Requires PyTorch (requirements-torch.txt).
Not thread-pinned on purpose: Table VI was generated at the default thread count (see paper, Data and Code Availability)."""
import json
from run_experiments import compare_dqn_vs_heuristics

out = {}
for seed in [42, 7, 19]:
    print(f"=== Table VI, training seed {seed} ===", flush=True)
    out[str(seed)] = compare_dqn_vs_heuristics(seed=seed)
    with open("table_vi_v2_all_seeds.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
print("DONE")
