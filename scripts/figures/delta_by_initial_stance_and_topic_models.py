"""
3 rows (topics) Ã— 6 columns (models).
x = initial stance (âˆ’2 â€¦ +2), y = mean Î” belief Â± 1 SD shaded band.
Lines: Human (grey) + one line per model.

Run from: HumanSimulationProject/
Output:   figures/plots/delta_by_initial_stance_and_topic_models.png / .pdf
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]

# â”€â”€ model color map (from colors.yaml) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
MODELS = {
    "GPT-5.2":                ("GPT-5.2",                "results/gpt5.2_results.xlsx",                                         _COLORS["GPT-5.2"]),
    "GPT-5-mini":             ("GPT-5-Mini",             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",            _COLORS["GPT-5-mini"]),
    "Claude Opus 4.6":        ("Claude-Opus-4.6",        "results/claude-opus-4-6_all_personas_results.xlsx",                  _COLORS["Claude Opus 4.6"]),
    "Gemini-3-flash":         ("Gemini-3-Flash-Preview", "results/gemini-3-flash-preview_all_personas_results.xlsx",           _COLORS["Gemini-3-flash"]),
    "Qwen3-32B":              ("Qwen3-32B",              "results/Qwen3_32B_all_personas_results.xlsx",                        _COLORS["Qwen3-32B"]),
    "Llama-3.3-70B-Instruct": ("Llama-3.3-70B-Instruct","results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",           _COLORS["Llama-3.3-70B-Instruct"]),
}

TOPIC_LABELS = {
    "UBI":         "UBI",
    "penalty":     "Penalty Shootouts",
    "weight_loss": "Weight Loss Drugs",
}
TOPICS       = ["UBI", "penalty", "weight_loss"]
INIT_STANCES = [-2, -1, 0, 1, 2]
HUMAN_COLOR = _yaml["human"]


# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    return df[["topic", "init_stance", "delta"]]


def load_llm(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = normalize(df)
    df = df.rename(columns={"init_belief": "init_stance"})
    df["delta"] = df["llm_new_belief"] - df["init_stance"]
    return df[["topic", "init_stance", "delta"]].dropna()


def mean_sd_by_init(df: pd.DataFrame, topic: str):
    sub   = df[df["topic"] == topic].groupby("init_stance")["delta"]
    means = sub.mean().reindex(INIT_STANCES)
    sds   = sub.std().reindex(INIT_STANCES)
    return means.values, sds.values


# â”€â”€ panel drawing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_panel(ax, human_df, llm_df, color, topic, show_ylabel, show_xlabel):
    # human reference
    h_mean, h_sd = mean_sd_by_init(human_df, topic)
    ax.plot(INIT_STANCES, h_mean, color=HUMAN_COLOR, lw=2,
            marker="o", markersize=5, zorder=4, label="Human")
    ax.fill_between(INIT_STANCES, h_mean - h_sd, h_mean + h_sd,
                    color=HUMAN_COLOR, alpha=0.15, zorder=1)

    # model line
    m, sd = mean_sd_by_init(llm_df, topic)
    ax.plot(INIT_STANCES, m, color=color, lw=2, linestyle="--",
            marker="o", markersize=4, zorder=3, label="LLM")
    ax.fill_between(INIT_STANCES, m - sd, m + sd,
                    color=color, alpha=0.12, zorder=1)

    ax.axhline(0, color="grey", lw=0.7, linestyle=":", zorder=0)
    ax.set_xticks(INIT_STANCES)
    ax.set_xticklabels([str(v) for v in INIT_STANCES], fontsize=7)
    ax.set_xlim(-2.4, 2.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)

    if show_xlabel:
        ax.set_xlabel("Initial Stance", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Mean Î” Â± 1 SD", fontsize=8)


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot():
    human_df    = load_human()
    model_keys  = list(MODELS.keys())
    n_models    = len(model_keys)

    llm_cache = {}
    for key in model_keys:
        _, path, _ = MODELS[key]
        llm_cache[key] = load_llm(path)

    fig, axes = plt.subplots(
        3, n_models,
        figsize=(3.2 * n_models, 10),
        sharex=True, sharey="row",
    )

    for row, topic in enumerate(TOPICS):
        for col, key in enumerate(model_keys):
            ax             = axes[row, col]
            label, _, color = MODELS[key]

            draw_panel(
                ax,
                human_df=human_df,
                llm_df=llm_cache[key],
                color=color,
                topic=topic,
                show_ylabel=(col == 0),
                show_xlabel=(row == 2),
            )

            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", pad=6, color=color)
            if col == 0:
                ax.set_ylabel(f"{TOPIC_LABELS[topic]}\n\nMean Î” Â± 1 SD", fontsize=9)

    # shared legend
    legend_handles = [
        Line2D([0], [0], color=HUMAN_COLOR, lw=2, marker="o", markersize=5, label="Human"),
    ] + [
        Line2D([0], [0], color=MODELS[k][2], lw=2, linestyle="--",
               marker="o", markersize=4, label=MODELS[k][0])
        for k in model_keys
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=n_models + 1,
        fontsize=8, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    out = "figures/plots/delta_by_initial_stance_and_topic_models"
    plt.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved {out}.svg / .png / .pdf")


plot()
