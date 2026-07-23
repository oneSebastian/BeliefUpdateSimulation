"""
Initial-stance and post-stance distribution panels for 6 models (own-initial-belief
ablation condition), one SVG/PNG/PDF per model Ã— stance â€” matching the visual style
of panel_claude.svg.

Run from: HumanSimulationProject/
Output:   figures/plots/distributions/initial_beliefs/panel_<model>_<init|post>.svg/png/pdf
"""

import sys

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

plt.rcParams.update({
    'font.size':          10,
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica'],
    'axes.labelsize':     10,
    'axes.titlesize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'lines.linewidth':    0.8,
    'axes.linewidth':     0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

OUT_DIR = "figures/plots/distributions/initial_beliefs"
os.makedirs(OUT_DIR, exist_ok=True)

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
MODEL_COLORS = _yaml["models"]
HUMAN_COLOR  = _yaml["human"]

MODELS = [
    ("Qwen3-32B",              "qwen",     "results/ablations/Qwen3-32B_own-initial-belief.xlsx"),
    ("Llama-3.3-70B-Instruct", "llama",    "results/ablations/Llama-3.3-70B-Instruct_own-initial-belief.xlsx"),
    ("GPT-5-mini",             "gpt5mini", "results/ablations/academic-ai-gpt-5-mini_own-initial-belief.xlsx"),
    ("GPT-5.2",                "gpt52",    "results/ablations/gpt-5.2_own-initial-belief.xlsx"),
    ("Gemini-3-flash",         "gemini",   "results/ablations/gemini-3-flash-preview_own-initial-belief.xlsx"),
    ("Claude Opus 4.6",        "claude",   "results/ablations/claude-opus-4-6_own-initial-belief.xlsx"),
]

# load_normalized_data already flips new_belief for negative formulations;
# init_stance must be flipped manually (same list used in ablation_stances.py).

LIKERT_VALUES = [-2, -1, 0, 1, 2]
LIKERT_LABELS = ["SD", "D", "N", "A", "SA"]


def load_ablation(path):
    df = load_normalized_data(path)
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "init_stance"] *= -1
    init = pd.to_numeric(df["init_stance"], errors="coerce").dropna().astype(int)
    post = pd.to_numeric(df["new_belief"],  errors="coerce").dropna().astype(int)
    return init, post


def load_human():
    df = load_human_normalized_data()
    init = pd.to_numeric(df["init_stance"], errors="coerce").dropna().astype(int)
    post = pd.to_numeric(df["new_belief"],  errors="coerce").dropna().astype(int)
    return init, post


def draw_panel(series, color, title, human_series, human_label, y_max):
    fig, ax = plt.subplots(figsize=(4.7, 4.5), constrained_layout=True)

    total  = len(series)
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    x      = np.arange(len(LIKERT_VALUES))

    bars = ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.4,
                  width=0.7, zorder=2, alpha=0.85)

    h_counts = [int((human_series == v).sum()) for v in LIKERT_VALUES]
    edges    = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(h_counts, edges, color=HUMAN_COLOR, lw=1.8, zorder=5, label=human_label)

    ax.set_title(title, fontsize=11, fontweight="normal", pad=8, color=color, loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(LIKERT_LABELS, fontsize=11.25)
    ax.set_ylabel("Frequency", fontsize=12.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.set_ylim(0, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11.25)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4, zorder=0)

    return fig


def main():
    human_init, human_post = load_human()
    print(f"Human: init n={len(human_init)}, post n={len(human_post)}")

    data_cache = []
    all_series = [human_init, human_post]
    for label, slug, path in MODELS:
        init, post = load_ablation(path)
        data_cache.append((label, slug, init, post))
        all_series.extend([init, post])
        print(f"{label}: init n={len(init)}, post n={len(post)}")

    global_max = max(
        max(int((s == v).sum()) for v in LIKERT_VALUES)
        for s in all_series
    )
    y_max = global_max + 5

    for label, slug, init, post in data_cache:
        color = MODEL_COLORS[label]

        fig = draw_panel(init, color, f"{label} Initial Stance",
                         human_init, "Human Initial Stance", y_max)
        stem = f"{OUT_DIR}/panel_{slug}_init"
        fig.savefig(f"{stem}.svg", format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved panel_{slug}_init")

        fig = draw_panel(post, color, f"{label} Post-Stance",
                         human_post, "Human Post-Stance", y_max)
        stem = f"{OUT_DIR}/panel_{slug}_post"
        fig.savefig(f"{stem}.svg", format='svg', bbox_inches='tight')
        plt.close(fig)
        print(f"Saved panel_{slug}_post")

    print(f"\nAll 12 panels saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
