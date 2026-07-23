import time
import itertools
import json
import string
import re
import unicodedata
import pandas as pd
import numpy as np
from scipy.stats import permutation_test
from belief_update_sim.config import DERIVED_DIR, ensure_output_dirs
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS
from belief_update_sim.response_parsing import parse_json_output as _parse_json_output


def perform_permutation_test(sample_1, sample_2):
    start = time.time()
    def statistic(x, y):
        return np.mean(x - y)
    observed_diff = np.mean(sample_1 - sample_2)
    # Perform paired permutation test
    N_TOTAL = 1_000_000
    CHUNK_SIZE = 100_000
    N_CHUNKS = N_TOTAL // CHUNK_SIZE
    count = 0
    obs = statistic(sample_1, sample_2)
    rng = np.random.default_rng(42)
    for _ in range(N_CHUNKS):
        res = permutation_test(
            (sample_1, sample_2),
            statistic,
            permutation_type="samples",random_state=rng,
            n_resamples=CHUNK_SIZE
        )
        count += np.sum(np.abs(res.null_distribution) >= np.abs(obs))

    p_value = count / N_TOTAL
    end = time.time()
    #print(f"Permutation test completed in {end - start:.2f} seconds.")
    #print(f"Observed mean difference (sample 1 - sample 2): {observed_diff:.4f}")
    #print(f"Permutation p-value: {p_value:.4f}")

    alpha = 0.0167
    reject = (res.pvalue < alpha)
    #if res.pvalue < alpha:
    #    print(f"Result: REJECT the null hypothesis (p = {res.pvalue:.4f} < \u03b1 = {alpha})")
    return observed_diff, p_value, reject


def parse_json_output(output):
    """Tuple-shaped adapter over the shared parser (belief_update_sim)."""
    parsed = _parse_json_output(output)
    if parsed is None:
        return None, None, None, None
    return (parsed["new_belief"], parsed["ranking"],
            parsed["reasoning"], parsed["general_public_stance"])


PUNCT_TRANSLATION = str.maketrans({
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
    "â€¦": "...",
})

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(PUNCT_TRANSLATION)
    return s


def process_row(row):
    new_belief, ranking, reasoning, general_public_stance = parse_json_output(row["raw_response"])
    if ranking is None:
         rank_1, rank_2, rank_3 = None, None, None
    else:
        if len(ranking) != 3:
            #msg = f"Expected ranking of length 3, got {len(ranking)}: {ranking}"
            #raise ValueError(msg)
            rank_1, rank_2, rank_3 = None, None, None
        else:
            rank_1, rank_2, rank_3 = ranking

    for col, new_val in zip(["llm_new_belief", "rank_1", "rank_2", "rank_3", "llm_general_public_stance", "llm_reasoning"], [new_belief, rank_1, rank_2, rank_3, general_public_stance, reasoning]):
        old_val = row[col]

        if pd.isna(old_val):
            row[col] = new_val
        else:
            if col == "llm_reasoning":
                assert normalize_text(old_val) == normalize_text(new_val), (
                    f"Mismatch in column {col}: "
                    f"existing={old_val}, parsed={new_val}"
                )
            else:
                assert old_val == new_val, (
                    f"Mismatch in column {col}: "
                    f"existing={old_val}, parsed={new_val}"
                )

    return row



def normalize_data(df):
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    negative_mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    cols_to_flip = ["llm_new_belief", "llm_general_public_stance"]
    df.loc[negative_mask, cols_to_flip] *= -1
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    return df


def load_from_path_normalized(path, i):
    df = pd.read_excel(path)
    df = df.apply(process_row, axis=1)
    df = normalize_data(df)
    df = df.rename(columns={"llm_new_belief": f"stance_{i}"})
    return df[["persona_id", "topic", f"stance_{i}"]]


def load_data(path1, path2="human"):
    paths = path1, path2
    data = []
    for i, path in enumerate(paths):
        if path == "human":
            df = pd.read_csv('results/merged_llm_participants_data_with_normalized_beliefs.csv')
            df = df[['persona_id', 'topic', 'final_belief_normalized']]
            df = df.rename(columns={"final_belief_normalized": f"stance_{i}"})
            data.append(df)
        else:
            data.append(load_from_path_normalized(path, i))

    # merge data
    df1, df2 = data
    key_cols = ["persona_id", "topic"]

    # 1. Same key coverage
    keys1 = set(map(tuple, df1[key_cols].to_numpy()))
    keys2 = set(map(tuple, df2[key_cols].to_numpy()))
    assert keys1 == keys2, "DataFrames do not contain the same (persona_id, topic) pairs"

    # 2. Same order (row-wise identical keys)
    #assert df1[key_cols].equals(df2[key_cols]), (
    #    "The (persona_id, topic) keys are not in the same order"
    #)

    # 3. Uniqueness of the composite key
    assert not df1.duplicated(key_cols).any(), "Duplicate (persona_id, topic) in df1"
    assert not df2.duplicated(key_cols).any(), "Duplicate (persona_id, topic) in df2"

    # TODO: given the identity in order the merge is unnecessary, but lets keep it in case we ever need to generalize
    # TODO: actually between "results/Qwen3_32B_all_personas_results.xlsx" and human data the order assertion failed!
    # 4. Merge
    merged = df1.merge(
        df2,
        on=key_cols,
        how="inner",
        validate="one_to_one"
    )

    full_data = len(merged)
    merged = merged.dropna(subset=["stance_0", "stance_1"])
    valid_data = len(merged)
    print(f"Dropped {full_data - valid_data} rows with missing values in stance columns")

    return merged["stance_0"].values, merged["stance_1"].values


def get_accuracy(ground_truth_sample, predictions):
    equal = ground_truth_sample == predictions
    count_equal = np.sum(equal)
    return count_equal / len(ground_truth_sample)


paths = {
    "human": "human",
    "qwen": "results/Qwen3_32B_all_personas_results.xlsx",
    "qwen_no-persona": "results/ablations/Qwen3-32B_no-persona.xlsx",
    "qwen_no-personality": "results/ablations/Qwen3-32B_no-personality.xlsx",
    "qwen_no-demographic": "results/ablations/Qwen3-32B_no-demographic.xlsx",
    "llama": "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
    "llama_no-persona": "results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx",
    "llama_no-personality": "results/ablations/Llama-3.3-70B-Instruct_no-personality.xlsx",
    "llama_no-demographic": "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx",
    "olmo-instruct": "results/Olmo-3.1-32B-Instruct_all_personas_results.xlsx",
    "olmo-instruct-sft": "results/Olmo-3.1-32B-Instruct-SFT_all_personas_results.xlsx",
    "olmo-instruct-dpo": "results/Olmo-3.1-32B-Instruct-DPO_all_personas_results.xlsx",
    "olmo-think": "results/Olmo-3-32B-Think_all_personas_results.xlsx",
    "olmo-think-sft": "results/Olmo-3-32B-Think-SFT_all_personas_results.xlsx",
    "olmo-think-dpo": "results/Olmo-3-32B-Think-DPO_all_personas_results.xlsx",
}

def collect_results(run_permutation_tests=True):
    results = []

    groups = [
        ["human", "olmo-instruct", "olmo-instruct-sft", "olmo-instruct-dpo"],
        ["human", "olmo-think", "olmo-think-sft", "olmo-think-dpo"]
    ]

    for group in groups:
        pairs = list(itertools.combinations(group, 2))
        for pair in pairs:
            sample_1, sample_2 = load_data(paths[pair[0]], paths[pair[1]])
            if run_permutation_tests:
                observed_diff, p_value, reject = perform_permutation_test(sample_1, sample_2)
            else:
                observed_diff, p_value, reject = None, None, None
            if pair[0] == "human":
                accuracy = get_accuracy(ground_truth_sample=sample_1, predictions=sample_2)
            elif pair[1] == "human":
                accuracy = get_accuracy(ground_truth_sample=sample_2, predictions=sample_1)
            else:
                accuracy = None
            data = {
                "sample_1": pair[0],
                "sample_2": pair[1],
                "observed_diff": observed_diff,
                "p_value": p_value,
                "reject": reject,
                "accuracy": accuracy,
            }
            results.append(data)
            print(data)

    df = pd.DataFrame(results)
    ensure_output_dirs()
    stem = f"post_training{'' if run_permutation_tests else '_no_permutation'}"
    df.to_csv(DERIVED_DIR / f"{stem}.csv", index=False)
    df.to_parquet(DERIVED_DIR / f"{stem}.parquet", index=False)


collect_results(run_permutation_tests=False)


        
