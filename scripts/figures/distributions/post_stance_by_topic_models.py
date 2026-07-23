"""
Post-stance distributions for all 6 models, broken down by topic.
6 rows (models) Ã— 3 columns (topics) for Nature Communications.

Run from: HumanSimulationProjectFigures/
Output:   figures/plots/distributions/post_stance_by_topic_models.svg / .png / .pdf
"""

import sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
import os

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica'],
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'lines.linewidth':    0.8,
    'axes.linewidth':     0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

os.makedirs("figures/plots/distributions", exist_ok=True)

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
MODEL_COLORS = _yaml["models"]
HUMAN_COLOR  = _yaml["human"]

MODELS = [
    ("GPT-5.2",                "results/gpt5.2_results.xlsx"),
    ("GPT-5-Mini",             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx"),
    ("Claude-Opus-4.6",        "results/claude-opus-4-6_all_personas_results.xlsx"),
    ("Gemini-3-Flash-Preview", "results/gemini-3-flash-preview_all_personas_results.xlsx"),
    ("Qwen3-32B",              "results/Qwen3_32B_all_personas_results.xlsx"),
    ("Llama-3.3-70B-Instruct", "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx"),
]

# Maps display labels to keys in colors.yaml (which uses legacy names)
COLOR_KEY = {
    "GPT-5.2":                "GPT-5.2",
    "GPT-5-Mini":             "GPT-5-mini",
    "Claude-Opus-4.6":        "Claude Opus 4.6",
    "Gemini-3-Flash-Preview": "Gemini-3-flash",
    "Qwen3-32B":              "Qwen3-32B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
}

TOPICS = ["UBI", "Penalty", "Weight Loss"]
TOPIC_LABELS = {
    "UBI":         "UBI",
    "Penalty":     "Penalty Shootouts",
    "Weight Loss": "Weight Loss Drugs",
}

LIKERT_VALUES = [-2, -1, 0, 1, 2]
LIKERT_LABELS = ["SD", "D", "N", "A", "SA"]


def get_topic(formulation):
    if not isinstance(formulation, str):
        return "Other"
    fl = formulation.lower()
    if "ubi" in fl or "universal basic income" in fl:
        return "UBI"
    if "penalty" in fl or "shootout" in fl:
        return "Penalty"
    if "ozempic" in fl or "weight loss" in fl:
        return "Weight Loss"
    return "Other"


def load_model_by_topic(path):
    df = load_normalized_data(path)
    df["_topic"] = df["statement_formulation"].apply(get_topic)
    return {
        t: pd.to_numeric(df.loc[df["_topic"] == t, "new_belief"],
                         errors="coerce").dropna().astype(int)
        for t in TOPICS
    }


def load_human_by_topic():
    df = load_human_normalized_data()
    df["_topic"] = df["statement_formulation"].apply(get_topic)
    return {
        t: pd.to_numeric(df.loc[df["_topic"] == t, "new_belief"],
                         errors="coerce").dropna().astype(int)
        for t in TOPICS
    }


def draw_panel(ax, series, color, human_series, y_max, show_ylabel):
    total    = len(series)
    counts   = [int((series == v).sum()) for v in LIKERT_VALUES]
    h_counts = [int((human_series == v).sum()) for v in LIKERT_VALUES]
    x        = np.arange(len(LIKERT_VALUES))

    bars = ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.4,
                  width=0.7, zorder=2, alpha=0.85)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100 if total > 0 else 0
        if cnt > 0:
            bar_top = bar.get_height()
            if bar_top + y_max * 0.10 > y_max:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_top - (y_max * 0.01),
                    f"{cnt}\n({pct:.0f}%)",
                    ha="center", va="top", fontsize=7,
                    fontweight="normal", linespacing=1.2,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar_top + (y_max * 0.01),
                    f"{cnt}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=7,
                    fontweight="normal", linespacing=1.2, clip_on=False,
                )

    edges = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(h_counts, edges, color=HUMAN_COLOR, lw=1.4, zorder=5, label="Human")

    ax.set_xticks(x)
    ax.set_xticklabels(LIKERT_LABELS, fontsize=9)

    if show_ylabel:
        ax.set_ylabel("Frequency", fontsize=9)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4, zorder=0)


def main():
    print("Loading data...")

    model_data = {}
    for label, path in MODELS:
        model_data[label] = load_model_by_topic(path)
        for t in TOPICS:
            print(f"  {label} / {t}: n={len(model_data[label][t])}")

    human_data = load_human_by_topic()
    for t in TOPICS:
        print(f"  Human / {t}: n={len(human_data[t])}")

    # Global y_max across all model Ã— topic panels and human overlays
    all_counts = [
        int((model_data[label][t] == v).sum())
        for label, _ in MODELS
        for t in TOPICS
        for v in LIKERT_VALUES
    ] + [
        int((human_data[t] == v).sum())
        for t in TOPICS
        for v in LIKERT_VALUES
    ]
    y_max = max(all_counts) + 5
    print(f"Global y_max: {y_max}")

    nrows = len(MODELS)
    ncols = len(TOPICS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 13))
    plt.subplots_adjust(left=0.14, right=0.97, top=0.95, bottom=0.06,
                        hspace=0.15, wspace=0.10)

    # Column headers
    for col, t in enumerate(TOPICS):
        axes[0, col].set_title(TOPIC_LABELS[t], fontsize=10, fontweight='bold', pad=10)

    for row, (label, _) in enumerate(MODELS):
        color = MODEL_COLORS[COLOR_KEY[label]]
        for col, t in enumerate(TOPICS):
            draw_panel(axes[row, col], model_data[label][t], color,
                       human_data[t], y_max, show_ylabel=(col == 0))

        # Colored model name to the left of each row
        axes[row, 0].text(
            -0.30, 0.5, label,
            transform=axes[row, 0].transAxes,
            fontsize=9, color=color, fontweight='bold',
            rotation=90, ha='center', va='center',
        )

    # Shared legend at the bottom
    handles, legend_labels = axes[-1, -1].get_legend_handles_labels()
    fig.legend(handles, legend_labels,
               loc='lower center', bbox_to_anchor=(0.5, 0.01),
               ncol=2, fontsize=9,
               frameon=True, framealpha=0.95, edgecolor='black', fancybox=False)

    out = "figures/plots/distributions/post_stance_by_topic_models"
    for fmt in ["svg", "png", "pdf"]:
        fig.savefig(f"{out}.{fmt}", format=fmt,
                    dpi=(300 if fmt != "svg" else None), bbox_inches='tight')
        print(f"Saved {out}.{fmt}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
