"""
3 rows (topics) Ã— 3 columns (models).
x = initial stance (âˆ’2 â€¦ +2), y = mean Î” belief Â± 1 SD shaded band.
Lines: Human + all 4 ablation conditions.

Run from: HumanSimulationProject/
Output:   figures/plots/delta_by_initial_stance_and_topic.png / .pdf
"""

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


MODELS = {
    "Qwen3-32B": {
        "base":           "results/Qwen3_32B_all_personas_results.xlsx",
        "no-persona":     "results/ablations/Qwen3-32B_no-persona.xlsx",
        "no-personality": "results/ablations/Qwen3-32B_no-personality.xlsx",
        "no-demographic": "results/ablations/Qwen3-32B_no-demographic.xlsx",
    },
    "Llama-3.3-70B-Instruct": {
        "base":           "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "no-persona":     "results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx",
        "no-personality": "results/ablations/Llama-3.3-70B-Instruct_no-personality.xlsx",
        "no-demographic": "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx",
    },
    "GPT-5-mini": {
        "base":           "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
        "no-persona":     "results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx",
        "no-personality": "results/ablations/academic-ai-gpt-5-mini_no-personality.xlsx",
        "no-demographic": "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx",
    },
}

TOPIC_LABELS = {
    "UBI":         "UBI",
    "penalty":     "Penalty Shootouts",
    "weight_loss": "Weight Loss Drugs",
}
TOPICS = ["UBI", "penalty", "weight_loss"]

CONDITION_LABELS = {
    "base":           "Base (full persona)",
    "no-persona":     "No persona",
    "no-personality": "No personality traits",
    "no-demographic": "No demographic info",
}

COLORS = {
    "base":           "#1a1a2e",
    "no-persona":     "#e74c3c",
    "no-personality": "#e67e22",
    "no-demographic": "#8e44ad",
}
with open("colors.yaml") as _f:
    HUMAN_COLOR = yaml.safe_load(_f)["human"]
INIT_STANCES = [-2, -1, 0, 1, 2]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_human() -> pd.DataFrame:
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"]       = df["final_belief_normalized"] - df["initial_belief_normalized"]
    df["init_stance"] = df["initial_belief_normalized"]
    return df[["persona_id", "topic", "init_stance", "delta"]]


def load_llm(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = normalize(df)
    df = df.rename(columns={"init_belief": "init_stance"})
    df["delta"] = df["llm_new_belief"] - df["init_stance"]
    return df[["persona_id", "topic", "init_stance", "delta"]].dropna()


def mean_sd_by_init(df: pd.DataFrame, topic: str):
    sub = df[df["topic"] == topic].groupby("init_stance")["delta"]
    means = sub.mean().reindex(INIT_STANCES)
    sds   = sub.std().reindex(INIT_STANCES)
    return means.values, sds.values


def draw_panel(ax, human_df, llm_by_cond, topic, show_ylabel, show_xlabel):
    # human
    h_mean, h_sd = mean_sd_by_init(human_df, topic)
    ax.plot(INIT_STANCES, h_mean, color=HUMAN_COLOR, lw=2,
            marker="o", markersize=5, zorder=4, label="Human")
    ax.fill_between(INIT_STANCES, h_mean - h_sd, h_mean + h_sd,
                    color=HUMAN_COLOR, alpha=0.15, zorder=1)

    # LLM conditions
    for cond, df in llm_by_cond.items():
        m, sd = mean_sd_by_init(df, topic)
        ax.plot(INIT_STANCES, m, color=COLORS[cond], lw=1.8, linestyle="--",
                marker="o", markersize=4, zorder=3, label=CONDITION_LABELS[cond])
        ax.fill_between(INIT_STANCES, m - sd, m + sd,
                        color=COLORS[cond], alpha=0.08, zorder=1)

    ax.axhline(0, color="grey", lw=0.7, linestyle=":", zorder=0)
    ax.set_xticks(INIT_STANCES)
    ax.set_xticklabels([str(v) for v in INIT_STANCES], fontsize=8)
    ax.set_xlim(-2.4, 2.4)
    ax.spines[["top", "right"]].set_visible(False)

    if show_xlabel:
        ax.set_xlabel("Initial Stance", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Mean Î” Belief Â± 1 SD", fontsize=9)


def plot():
    human_df = load_human()
    llm_cache = {
        model: {cond: load_llm(path) for cond, path in paths.items()}
        for model, paths in MODELS.items()
    }

    model_names = list(MODELS.keys())

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle(
        "Mean Belief Change by Initial Stance and Topic  (Â± 1 SD)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for row, topic in enumerate(TOPICS):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]
            draw_panel(
                ax,
                human_df=human_df,
                llm_by_cond=llm_cache[model_name],
                topic=topic,
                show_ylabel=(col == 0),
                show_xlabel=(row == 2),
            )

            if row == 0:
                ax.set_title(model_name, fontsize=10, fontweight="bold", pad=6)
            if col == 0:
                ax.set_ylabel(
                    f"{TOPIC_LABELS[topic]}\n\nMean Î” Â± 1 SD",
                    fontsize=9,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    plt.savefig("figures/plots/delta_by_initial_stance_and_topic.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/delta_by_initial_stance_and_topic.pdf", bbox_inches="tight")
    print("Saved figures/plots/delta_by_initial_stance_and_topic.png")


plot()
