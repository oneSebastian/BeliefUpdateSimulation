import time
import pandas as pd
import numpy as np
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


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

def normalize_data(df):
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    negative_mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    cols_to_flip = ["llm_new_belief", "llm_general_public_stance", "init_belief"]
    df.loc[negative_mask, cols_to_flip] *= -1
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    return df


def load_from_path_normalized(path, i):
    df = pd.read_excel(path)
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
        elif path == "human-initial-belief":
            df = pd.read_csv('results/merged_llm_participants_data_with_normalized_beliefs.csv')
            df = df[['persona_id', 'topic', 'initial_belief_normalized']]
            df = df.rename(columns={"initial_belief_normalized": f"stance_{i}"})
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

    return merged["stance_0"].values, merged["stance_1"].values


def get_accuracy(ground_truth_sample, predictions):
    equal = ground_truth_sample == predictions
    count_equal = np.sum(equal)
    return count_equal / len(ground_truth_sample)


def collect_data():
    # compare human pre and post stance
    sample_1, sample_2 = load_data(path1="human-initial-belief", path2="human")
    observed_diff, p_value, reject = perform_permutation_test(sample_1, sample_2)
    print(f"Human initial vs final belief: observed_diff={observed_diff:.4f}, p_value={p_value:.4f}, reject={reject}, accuracy={get_accuracy(ground_truth_sample=sample_2, predictions=sample_1):.4f}")

    paths = {
        "qwen": "results/ablations/Qwen3-32B_own-initial-belief.xlsx",
        "llama": "results/ablations/Llama-3.3-70B-Instruct_own-initial-belief.xlsx"
    }
    for name, path in paths.items():
        sample_1, sample_2 = load_data(path1=path, path2="human")
        observed_diff, p_value, reject = perform_permutation_test(sample_1, sample_2)
        print(f"{name}: observed_diff={observed_diff:.4f}, p_value={p_value:.4f}, reject={reject}, accuracy={get_accuracy(ground_truth_sample=sample_2, predictions=sample_1):.4f}")

        # compare human pre and LLM pre stance
        sample_1, sample_2 = load_data(path1=path, path2="human-initial-belief")
        observed_diff, p_value, reject = perform_permutation_test(sample_1, sample_2)
        print(f"Human initial vs {name} initial stance: observed_diff={observed_diff:.4f}, p_value={p_value:.4f}, reject={reject}, accuracy={get_accuracy(ground_truth_sample=sample_2, predictions=sample_1):.4f}")


def plot_likert(ax, data, title, color):
    likert_values = [-2, -1, 0, 1, 2]
    likert_labels = [
        "Strongly Disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly Agree"
    ]
    counts = [np.sum(data == v) for v in likert_values]
    total = len(data)
    percentages = [c / total * 100 for c in counts]

    bars = ax.bar(likert_labels, counts, color=color, edgecolor="black")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.tick_params(axis="x", rotation=25)

    # Annotate counts + percentages
    for bar, c, p in zip(bars, counts, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{c}\n({p:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )


def descriptive_results():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Distribution of LLM initial stances and post-stances",
        fontsize=16,
        fontweight="bold"
    )

    for i, model in enumerate(["Qwen3-32B", "Llama-3.3-70B-Instruct"]):
        df = pd.read_excel(f"results/ablations/{model}_own-initial-belief.xlsx")
        df = normalize_data(df)
        print(f"{model} initial stance mean: {df['init_belief'].mean():.4f}, std: {df['init_belief'].std():.4f}")
        print(f"{model} post stance mean: {df['llm_new_belief'].mean():.4f}, std: {df['llm_new_belief'].std():.4f}")
        plot_likert(axes[i, 0], df['init_belief'],  f"{model} Initial Stance Distribution",  "#b22222")
        plot_likert(axes[i, 1], df['llm_new_belief'],  f"{model} Post-Stance Distribution",  "#b22222")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/ablations/results/own_belief_distribution.svg", bbox_inches="tight")

#"llm_new_belief", "llm_general_public_stance", "init_belief"


#collect_data()
descriptive_results()
