"""
Plot 2: Between-persona spread per condition and topic.
Computes the SD of Î” *across* persona_ids within each (topic, condition).
If persona info creates individual diversity in outputs, base should show
larger between-persona SD than no-persona.

Layout: 1Ã—3 panels (one per model).
Within each: grouped bars by topic (3 groups Ã— 4 conditions).
Human between-persona SD shown as horizontal reference lines per topic.

Run from: HumanSimulationProject/
Output:   claude/persona_influence_between_spread.png
"""

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


MODELS = {
    "GPT-5-mini": {
        "Base":           "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
        "No demographic": "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx",
        "No personality": "results/ablations/academic-ai-gpt-5-mini_no-personality.xlsx",
        "No persona":     "results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx",
    },
    "Qwen3-32B": {
        "Base":           "results/Qwen3_32B_all_personas_results.xlsx",
        "No demographic": "results/ablations/Qwen3-32B_no-demographic.xlsx",
        "No personality": "results/ablations/Qwen3-32B_no-personality.xlsx",
        "No persona":     "results/ablations/Qwen3-32B_no-persona.xlsx",
    },
    "Llama-3.3-70B": {
        "Base":           "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "No demographic": "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx",
        "No personality": "results/ablations/Llama-3.3-70B-Instruct_no-personality.xlsx",
        "No persona":     "results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx",
    },
}

TOPICS = ["UBI", "penalty", "weight_loss"]
TOPIC_LABELS = {"UBI": "UBI", "penalty": "Penalty", "weight_loss": "Wt. Loss"}
CONDITIONS = ["Base", "No demographic", "No personality", "No persona"]

COND_COLORS = {
    "Base":           "#1a1a2e",
    "No demographic": "#8e44ad",
    "No personality": "#e67e22",
    "No persona":     "#e74c3c",
}
TOPIC_HATCH = {"UBI": "", "penalty": "//", "weight_loss": ".."}
with open("colors.yaml") as _f:
    HUMAN_COLOR = yaml.safe_load(_f)["human"]


def normalize(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_llm(path):
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df[["persona_id", "topic", "delta"]].dropna()


def load_human():
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    return df[["persona_id", "topic", "delta"]]


def between_persona_sd(df, topic):
    sub = df[df["topic"] == topic]
    return sub.groupby("persona_id")["delta"].mean().std()


def plot():
    human_df = load_human()
    human_sds = {t: between_persona_sd(human_df, t) for t in TOPICS}

    n_cond   = len(CONDITIONS)
    n_topics = len(TOPICS)
    group_width = 0.8
    bar_w = group_width / n_cond
    offsets = np.linspace(-(group_width - bar_w) / 2,
                           (group_width - bar_w) / 2, n_cond)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Between-Persona Spread of Î” Belief per Topic and Condition\n"
        "(SD across persona_ids â€” higher = more individual diversity in outputs)",
        fontsize=11, fontweight="bold", y=1.03,
    )

    x = np.arange(n_topics)

    for ax, (model, paths) in zip(axes, MODELS.items()):
        # human reference lines
        for ti, topic in enumerate(TOPICS):
            ax.plot([ti - 0.45, ti + 0.45], [human_sds[topic]] * 2,
                    color=HUMAN_COLOR, lw=2, linestyle="--", zorder=5)

        for ci, cond in enumerate(CONDITIONS):
            df = load_llm(paths[cond])
            sds = [between_persona_sd(df, t) for t in TOPICS]
            ax.bar(x + offsets[ci], sds, width=bar_w,
                   color=COND_COLORS[cond], label=cond,
                   edgecolor="white", zorder=2)

        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([TOPIC_LABELS[t] for t in TOPICS], fontsize=9)
        ax.set_xlabel("Topic", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("SD of mean Î” across personas", fontsize=9)

    # legend: conditions + human reference
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color=HUMAN_COLOR, lw=2, linestyle="--"))
    labels.append("Human reference")
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    plt.savefig("figures/plots/persona_influence_between_spread.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/persona_influence_between_spread.pdf", bbox_inches="tight")
    print("Saved figures/plots/persona_influence_between_spread.png")


plot()
