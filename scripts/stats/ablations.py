import json
from pathlib import Path

import pandas as pd

from belief_update_sim.config import STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.permutation_test import permutation_test_on_paths

with open(Path(__file__).with_name("data_paths.json")) as f:
    data_paths = json.load(f)

ensure_output_dirs()


def run_reduced_persona_permutation_tests():
    results = pd.DataFrame(columns=["LLM", "Condition", "Observed Diff", "p-value", "Reject Null"])
    for llm, paths in data_paths["ablation"].items():
        base_condition_path = data_paths["llm"][llm]
        print(f"Running permutation test for {llm} base condition...")
        observed_diff, p_value, reject = permutation_test_on_paths(data_paths["human"], base_condition_path)
        results.loc[len(results)] = [llm, "base", observed_diff, p_value, reject]
        results.to_parquet(f"{STATS_OUTPUT_DIR}/ablation_permutation_test_results.parquet")

        for condition, path in paths.items():
            print(f"Running permutation test for {llm} {condition} condition...")
            observed_diff, p_value, reject = permutation_test_on_paths(data_paths["human"], path)
            results.loc[len(results)] = [llm, condition, observed_diff, p_value, reject]
            results.to_parquet(f"{STATS_OUTPUT_DIR}/ablation_permutation_test_results.parquet")
            results.to_json(f"{STATS_OUTPUT_DIR}/ablation_permutation_test_results.json", orient='records', indent=2)


    print(results.to_string())


def within_model_permutation_tests():
    results = pd.DataFrame(columns=["LLM", "Condition 1", "Condition 2", "Observed Diff", "p-value", "Reject Null"])
    for llm, paths in data_paths["ablation"].items():
        base_condition_path = data_paths["llm"][llm]
        for condition, path in paths.items():
            print(f"Running within-model permutation test for {llm} {condition} vs base condition...")
            observed_diff, p_value, reject = permutation_test_on_paths(base_condition_path, path)
            results.loc[len(results)] = [llm, "base", condition, observed_diff, p_value, reject]
            results.to_parquet(f"{STATS_OUTPUT_DIR}/within_model_permutation_test_results.parquet")
            results.to_json(f"{STATS_OUTPUT_DIR}/within_model_permutation_test_results.json", orient='records', indent=2)

    print(results.to_string())


def own_initial_beliefs_permutation_tests():
    results = pd.DataFrame(columns=["LLM", "Observed Diff", "p-value", "Reject Null"])
    for llm, path in data_paths["initial_beliefs"].items():
        print(f"Running own-initial-beliefs permutation test for {llm}...")
        observed_diff, p_value, reject = permutation_test_on_paths(data_paths["human"], path)
        results.loc[len(results)] = [llm, observed_diff, p_value, reject]
        results.to_parquet(f"{STATS_OUTPUT_DIR}/own_initial_beliefs_permutation_test_results.parquet")
        results.to_json(f"{STATS_OUTPUT_DIR}/own_initial_beliefs_permutation_test_results.json", orient='records', indent=2)

    print(results.to_string())



run_reduced_persona_permutation_tests()
#within_model_permutation_tests()
#own_initial_beliefs_permutation_tests()
