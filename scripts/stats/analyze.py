import json
from pathlib import Path

from belief_update_sim.permutation_test import permutation_test_on_paths

with open(Path(__file__).with_name("data_paths.json")) as f:
    data_paths = json.load(f)

results = {}
for llm, path in data_paths["llm"].items():
    result = permutation_test_on_paths(data_paths["human"], path)
    results[llm] = result

print("Permutation Test Results:")
for llm, (observed_diff, p_value, reject) in results.items():
    print(f"{llm}: Observed Diff = {observed_diff:.4f}, p-value = {p_value:.4e}, Reject Null = {reject}")
