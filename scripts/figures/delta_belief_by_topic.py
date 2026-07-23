"""
Belief change (Î” = post âˆ’ initial) distribution by topic.
3 rows (topics: UBI, penalty, weight_loss) Ã— 3 columns (Qwen, Llama, GPT-5-mini).
Each panel overlays human vs all ablation conditions.

Run from: HumanSimulationProject/
Output:   claude/delta_belief_by_topic.png
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
    "GPT-5-Mini": {
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

CONDITION_LABELS = {
    "base":           "Base (full persona)",
    "no-persona":     "No persona",
    "no-personality": "No personality traits",
    "no-demographic": "No demographic info",
}

COLORS = {
    "base":           "#4dac26",   # forest green
    "no-persona":     "#d01c8b",   # magenta
    "no-personality": "#0571b0",   # dark blue
    "no-demographic": "#ca0020",   # crimson
}

with open("colors.yaml") as _f:
    HUMAN_COLOR = yaml.safe_load(_f)["human"]
TOPICS = ["UBI", "penalty", "weight_loss"]

DELTA_BINS  = np.arange(-4.5, 5.5, 1.0)
DELTA_TICKS = np.arange(-4, 5)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_human() -> pd.DataFrame:
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    return df


def load_llm(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df[["persona_id", "topic", "delta"]].dropna()


def proportions(delta: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(delta, bins=DELTA_BINS)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


def draw_panel(ax, human_delta, llm_data_by_cond, topic, model_name, show_ylabel, show_xlabel):
    conds      = list(llm_data_by_cond.keys())
    n_conds    = len(conds)
    bar_w      = 0.17                          # width of each small bar
    step       = bar_w + 0.02                  # centre-to-centre spacing
    offsets    = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * step

    # grouped ablation bars
    for offset, cond in zip(offsets, conds):
        props = proportions(llm_data_by_cond[cond])
        ax.bar(DELTA_TICKS + offset, props, width=bar_w,
               color=COLORS[cond], edgecolor="white", linewidth=0.3,
               zorder=2, label=CONDITION_LABELS[cond])

    # human baseline: grey step outline
    h_props = proportions(human_delta)
    edges   = np.append(DELTA_TICKS - 0.5, DELTA_TICKS[-1] + 0.5)
    ax.stairs(h_props, edges, color=HUMAN_COLOR, lw=1.8, zorder=3, label="Human")

    ax.axvline(0, color="grey", lw=0.7, linestyle=":", zorder=0)
    ax.set_xticks(DELTA_TICKS)
    ax.set_xticklabels([str(v) for v in DELTA_TICKS], fontsize=12)
    ax.set_xlim(-4.7, 4.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)

    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=14)
    if show_xlabel:
        ax.set_xlabel("Î” Belief  (post âˆ’ initial)", fontsize=14)


def plot():
    human_df = load_human()

    # pre-load all LLM data
    llm_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for model_name, paths in MODELS.items():
        llm_cache[model_name] = {
            cond: load_llm(path) for cond, path in paths.items()
        }

    fig, axes = plt.subplots(
        3, 3, figsize=(15, 12),
        sharey=False,   # each topic may have different y scale
        sharex=True,
    )

    model_names = list(MODELS.keys())

    for row, topic in enumerate(TOPICS):
        for col, model_name in enumerate(model_names):
            ax = axes[row, col]

            human_topic = human_df.loc[human_df["topic"] == topic, "delta"].values
            llm_by_cond = {
                cond: df.loc[df["topic"] == topic, "delta"].values
                for cond, df in llm_cache[model_name].items()
            }

            draw_panel(
                ax,
                human_delta=human_topic,
                llm_data_by_cond=llm_by_cond,
                topic=topic,
                model_name=model_name,
                show_ylabel=(col == 0),
                show_xlabel=(row == 2),
            )

            if row == 0:
                ax.set_title(model_name, fontsize=14, fontweight="bold", pad=6)

            if col == 0:
                ax.set_ylabel(
                    f"{TOPIC_LABELS[topic]}\n\nProportion",
                    fontsize=14,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        fontsize=13,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    out = "figures/plots/delta_belief_by_topic"
    plt.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved {out}.svg / .png / .pdf")


plot()
