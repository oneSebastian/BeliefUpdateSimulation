"""
Distribution of LLM initial stances and post-stances â€” own-initial-stance condition.
Extends the original 2-row figure (Qwen, Llama) with a third row for GPT-5-mini.

Run from: HumanSimulationProject/
Output:   figures/plots/distributions/ablation_stances_extended.png / .pdf / .svg
"""

import sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
import os
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# Fonts are pre-scaled for the merged output (ablation panel â‰ˆ 91Ã—96 mm).
# The merge script uses the SVG at near-native scale, so 8 pt here â†’ 8 pt printed.
_F = 8   # base font size (pt) for Nature Communications
plt.rcParams.update({
    'font.size':          _F,
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica'],
    'axes.labelsize':     _F,
    'axes.titlesize':     _F + 1,
    'xtick.labelsize':    _F - 1,
    'ytick.labelsize':    _F - 1,
    'legend.fontsize':    _F - 1,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
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

# â”€â”€ config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MODELS = [
    ("Qwen3-32B",              "results/ablations/Qwen3-32B_own-initial-belief.xlsx"),
    ("Llama-3.3-70B-Instruct", "results/ablations/Llama-3.3-70B-Instruct_own-initial-belief.xlsx"),
    ("GPT-5-mini",             "results/ablations/academic-ai-gpt-5-mini_own-initial-belief.xlsx"),
]


LIKERT_VALUES = [-2, -1, 0, 1, 2]
LIKERT_LABELS = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]

# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_data(path):
    df = load_normalized_data(path)
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "init_stance"] *= -1
    return df


def load_human():
    df = load_human_normalized_data()
    init = pd.to_numeric(df["init_stance"], errors="coerce").dropna().astype(int)
    post = pd.to_numeric(df["new_belief"],  errors="coerce").dropna().astype(int)
    return init, post


def draw_human_step(ax, series, label):
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    edges  = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(counts, edges, color=HUMAN_COLOR, lw=1.8, zorder=5, label=label)


def count_distribution(series):
    s      = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    total  = len(s)
    counts = [int((s == v).sum()) for v in LIKERT_VALUES]
    return counts, total


def draw_bars(ax, counts, total, title, color, show_ylabel=True):
    x    = np.arange(len(LIKERT_VALUES))
    bars = ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.4,
                  width=0.7, zorder=2, alpha=0.85)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100 if total > 0 else 0
        if cnt > 0:
            label_y = bar.get_height() + (max(counts) * 0.02)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{cnt}\n({pct:.0f}%)",
                ha="center", va="bottom", fontsize=_F - 1,
                fontweight="normal", linespacing=1.2,
            )

    ax.set_title(title, fontsize=_F, fontweight="normal", pad=4, color=color, loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(LIKERT_LABELS, fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Frequency", fontsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4, zorder=0)


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    n_models = len(MODELS)
    # Target output size: ablation panel â‰ˆ 91 mm Ã— 96 mm (3.58" Ã— 3.78")
    fig, axes = plt.subplots(
        n_models, 2,
        figsize=(3.58, 3.78),
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.97,
                        bottom=0.14, hspace=0.70, wspace=0.25)

    human_init, human_post = load_human()

    global_max = 0
    for series in [human_init, human_post]:
        global_max = max(global_max, max((series == v).sum() for v in LIKERT_VALUES))
    data_cache = []
    for label, path in MODELS:
        df = load_data(path)
        init_counts, init_total = count_distribution(df["init_stance"])
        post_counts, post_total = count_distribution(df["new_belief"])
        data_cache.append((label, init_counts, init_total, post_counts, post_total))
        global_max = max(global_max, max(init_counts), max(post_counts))

    y_max = global_max * 1.28

    for row_idx, (label, init_counts, init_total, post_counts, post_total) in enumerate(data_cache):
        ax_init = axes[row_idx, 0]
        ax_post = axes[row_idx, 1]
        color   = MODEL_COLORS[label]

        draw_bars(ax_init, init_counts, init_total,
                  f"{label} Initial Stance", color=color, show_ylabel=True)
        draw_human_step(ax_init, human_init, label="Human Initial Stance")

        draw_bars(ax_post, post_counts, post_total,
                  f"{label} Post-Stance", color=color, show_ylabel=False)
        draw_human_step(ax_post, human_post, label="Human Post-Stance")

        ax_init.set_ylim(0, y_max)
        ax_post.set_ylim(0, y_max)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, frameon=True, framealpha=0.95,
               edgecolor='black', fancybox=False,
               bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure)

    out = "figures/plots/distributions/ablation_stances_extended"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.svg", bbox_inches="tight")
    print(f"Saved {out}.png / .pdf / .svg")


main()
