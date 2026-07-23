"""
Point 2: Does initial stance moderate persuasion?
         People/LLMs with extreme initial stances should shift less (ceiling/floor effect).

Visualisation: 1Ã—3 grid (one panel per model).
x = initial stance (âˆ’2 â€¦ +2), y = mean Î” belief.
Lines: Human + all 4 ablation conditions.
Error bars = Â±1 SE across (persona_id, topic) observations.

Run from: HumanSimulationProject/
Output:   figures/plots/initial_stance_moderator.png
"""

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


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

# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ stats â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def mean_se_by_init(df: pd.DataFrame):
    grouped = df.groupby("init_stance")["delta"]
    means = grouped.mean()
    ses   = grouped.sem()
    return means.reindex(INIT_STANCES), ses.reindex(INIT_STANCES)


# â”€â”€ plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot():
    human_df  = load_human()
    h_mean, h_se = mean_se_by_init(human_df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    fig.suptitle(
        "Initial Stance as Moderator of Persuasion  (mean Î” Â± 1 SE)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (model_name, paths) in zip(axes, MODELS.items()):
        # human reference
        ax.errorbar(
            INIT_STANCES, h_mean, yerr=h_se,
            color=HUMAN_COLOR, lw=2, marker="o", markersize=6,
            capsize=3, label="Human", zorder=4,
        )

        for cond, path in paths.items():
            llm_df = load_llm(path)
            m, se  = mean_se_by_init(llm_df)
            ax.errorbar(
                INIT_STANCES, m, yerr=se,
                color=COLORS[cond], lw=1.8, marker="o", markersize=5,
                capsize=3, linestyle="--", label=CONDITION_LABELS[cond], zorder=3,
            )

        ax.axhline(0, color="grey", lw=0.8, linestyle=":", zorder=0)
        ax.set_xticks(INIT_STANCES)
        ax.set_xticklabels([str(v) for v in INIT_STANCES])
        ax.set_xlabel("Initial Stance", fontsize=10)
        ax.set_title(model_name, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8.5, framealpha=0.9)

    axes[0].set_ylabel("Mean Î” Belief  (post âˆ’ initial)", fontsize=10)

    plt.tight_layout()
    plt.savefig("figures/plots/initial_stance_moderator.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/initial_stance_moderator.pdf", bbox_inches="tight")
    print("Saved figures/plots/initial_stance_moderator.png")


plot()