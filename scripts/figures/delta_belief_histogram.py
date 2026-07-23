"""
Belief change (Î” = post âˆ’ initial) distribution comparison.
Humans vs Qwen3-32B and Llama-3.3-70B-Instruct across ablation conditions.

Run from: HumanSimulationProject/
Output:   claude/delta_belief_histogram.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import matplotlib.patches as mpatches
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


CONDITIONS = {
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

DELTA_BINS = np.arange(-4.5, 5.5, 1.0)   # bin edges
DELTA_TICKS = np.arange(-4, 5)            # -4 â€¦ +4


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"] *= -1
    return df


def load_human() -> pd.DataFrame:
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    return df


def load_llm_delta(path: str) -> np.ndarray:
    df = pd.read_excel(path)
    df = normalize(df)
    # init_belief after normalization == initial_belief_normalized
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df["delta"].dropna().values


def proportions(delta: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(delta, bins=DELTA_BINS)
    return counts / counts.sum()


def plot():
    human_df = load_human()
    human_props = proportions(human_df["delta"].values)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), sharey=True)
    fig.suptitle(
        "Belief Change Distribution  (Î” = post-stance âˆ’ initial stance)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, (model_name, paths) in zip(axes, CONDITIONS.items()):
        # â”€â”€ human bars (background) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ax.bar(
            DELTA_TICKS,
            human_props,
            width=0.85,
            color=HUMAN_COLOR,
            alpha=0.25,
            zorder=1,
            label="_nolegend_",
        )
        ax.step(
            np.append(DELTA_TICKS - 0.5, DELTA_TICKS[-1] + 0.5),
            np.append(human_props, human_props[-1]),
            where="post",
            color=HUMAN_COLOR,
            lw=1.8,
            zorder=2,
            label="Human",
        )

        # â”€â”€ LLM condition lines â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for cond, path in paths.items():
            props = proportions(load_llm_delta(path))
            ax.plot(
                DELTA_TICKS,
                props,
                color=COLORS[cond],
                lw=2,
                marker="o",
                markersize=5,
                zorder=3,
                label=CONDITION_LABELS[cond],
            )

        # â”€â”€ styling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ax.set_title(model_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Î” Belief  (post âˆ’ initial)", fontsize=10)
        ax.set_xticks(DELTA_TICKS)
        ax.set_xticklabels([str(v) for v in DELTA_TICKS])
        ax.set_xlim(-4.7, 4.7)
        ax.axvline(0, color="grey", lw=0.8, linestyle=":", zorder=0)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Proportion", fontsize=10)
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("figures/plots/delta_belief_histogram.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/delta_belief_histogram.pdf", bbox_inches="tight")
    print("Saved figures/plots/delta_belief_histogram.png")


plot()
