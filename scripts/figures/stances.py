import time
import pandas as pd
import numpy as np
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


def normalize_data(df):
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    negative_mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    cols_to_flip = ["llm_new_belief", "llm_general_public_stance", "init_belief"]
    df.loc[negative_mask, cols_to_flip] *= -1
    # print(df[["persona_id", "statement_formulation", "llm_new_belief", "llm_general_public_stance"]].head(20))
    return df


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


def human_results():
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(
        "Distribution of human pre- and post-stances",
        fontsize=16,
        fontweight="bold"
    )

    #for i, model in enumerate(["Qwen3-32B", "Llama-3.3-70B-Instruct"]):
    df = pd.read_csv(f"results/merged_llm_participants_data_with_normalized_beliefs.csv")
    plot_likert(axes[0], df['initial_belief_normalized'],  f"Pre-Stance Distribution",  "#6baed6")
    plot_likert(axes[1], df['final_belief_normalized'],  f"Post-Stance Distribution",  "#08519c")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/ablations/results/human_belief_distribution.svg", bbox_inches="tight")


def llm_results():
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle(
        "Distribution of LLM post-stances",
        fontsize=16,
        fontweight="bold"
    )

    paths = {
        "Gpt5.2": "results/gpt5.2_results.xlsx",
        "Qwen3_32B": "results/Qwen3_32B_all_personas_results.xlsx",
        "Llama-3.3-70B-Instruct": "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
    }

    colours = {
        "Gpt5.2": "#e57373",
        "Qwen3_32B": "#b22222",
        "Llama-3.3-70B-Instruct": "#7f1d1d",
    }

    i = 0
    for model, path in paths.items():
        df = pd.read_excel(path)
        df = normalize_data(df)
        plot_likert(axes[i], df['llm_new_belief'],  f"{model} Post-Stance Distribution",  colours[model])
        i += 1

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/ablations/results/LLM_belief_distribution.svg", bbox_inches="tight")


human_results()
llm_results()
