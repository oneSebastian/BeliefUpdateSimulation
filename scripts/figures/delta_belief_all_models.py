"""
Î” belief distributions for all models, split by topic.
1 row Ã— 3 columns (one panel per topic).
Each panel overlays Human + all LLMs as step lines.

Run from: HumanSimulationProject/
Output:   claude/delta_belief_all_models.png
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_MODEL_COLORS = _yaml["models"]
HUMAN_COLOR = _yaml["human"]

MODELS = {
    "Human":                  "human",
    "GPT-5.2":                "results/gpt5.2_results.xlsx",
    "GPT-5-mini":             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
    "Claude Opus 4.6":        "results/claude-opus-4-6_all_personas_results.xlsx",
    "Gemini-3-flash":         "results/gemini-3-flash-preview_all_personas_results.xlsx",
    "Qwen3-32B":              "results/Qwen3_32B_all_personas_results.xlsx",
    "Llama-3.3-70B-Instruct": "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
}

COLORS = {"Human": HUMAN_COLOR, **_MODEL_COLORS}

TOPIC_LABELS = {"UBI": "UBI", "penalty": "Penalty Shootouts", "weight_loss": "Weight Loss Drugs"}
TOPICS       = ["UBI", "penalty", "weight_loss"]
DELTA_BINS   = np.arange(-4.5, 5.5, 1.0)
DELTA_TICKS  = np.arange(-4, 5)


def normalize(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load(spec):
    if spec == "human":
        df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
        df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
        return df[["topic", "delta"]]
    df = pd.read_excel(spec)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df[["topic", "delta"]].dropna()


def proportions(delta):
    counts, _ = np.histogram(delta, bins=DELTA_BINS)
    return counts / counts.sum() if counts.sum() > 0 else counts.astype(float)


def plot():
    # load all data once
    data = {label: load(spec) for label, spec in MODELS.items()}

    llm_labels = [l for l in MODELS if l != "Human"]
    human_df   = data["Human"]

    fig, axes = plt.subplots(6, 3, figsize=(13, 18), sharey=False, sharex=True)
    fig.suptitle(
        "Belief Change Distribution (Î” = post âˆ’ initial)\nHuman reference (bars) vs each LLM (line)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for row, model_label in enumerate(llm_labels):
        for col, topic in enumerate(TOPICS):
            ax = axes[row, col]

            # human bars
            h_props = proportions(human_df[human_df["topic"] == topic]["delta"].values)
            ax.bar(DELTA_TICKS, h_props, width=0.85,
                   color=COLORS["Human"], alpha=0.25, zorder=1)

            # LLM line
            m_props = proportions(data[model_label][data[model_label]["topic"] == topic]["delta"].values)
            ax.plot(DELTA_TICKS, m_props,
                    color=COLORS[model_label], lw=2, marker="o", markersize=4,
                    zorder=3)

            ax.axvline(0, color="grey", lw=0.7, linestyle=":", zorder=0)
            ax.set_xticks(DELTA_TICKS)
            ax.set_xlim(-4.7, 4.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelsize=7)

            if col == 0:
                ax.set_ylabel(model_label, fontsize=8.5, fontweight="bold",
                              color=COLORS[model_label])
            if row == 0:
                ax.set_title(TOPIC_LABELS[topic], fontsize=10, fontweight="bold")
            if row == 5:
                ax.set_xlabel("Î” Belief", fontsize=8)

    # shared legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(color=COLORS["Human"], alpha=0.4, label="Human"),
    ] + [
        Line2D([0], [0], color=COLORS[l], lw=2, marker="o", markersize=5, label=l)
        for l in llm_labels
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig("figures/plots/delta_belief_all_models.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/delta_belief_all_models.pdf", bbox_inches="tight")
    print("Saved figures/plots/delta_belief_all_models.png")


plot()
