"""
Post-stance distributions for all 6 models, pooled across topics.
2 rows Ã— 3 columns (one panel per model) - Full page width for Nature Communications.

Run from: HumanSimulationProject/
Output:   figures/plots/distributions/post_stance_models.png / .pdf / .eps / .svg
"""

import sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
import os

# Set Nature Communications style parameters for FULL PAGE (not two-column)
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Create output directory
os.makedirs("figures/plots/distributions", exist_ok=True)

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]
HUMAN_COLOR = _yaml["human"]

# â”€â”€ model color map (from colors.yaml) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Layout: 2 rows, 3 columns - Order: Row1: GPT-5.2, GPT-5-mini, Gemini-3-flash
#                                      Row2: Claude Opus 4.6, Qwen3-32B, Llama-3.3-70B-Instruct
MODELS = {
    "GPT-5.2":                ("GPT-5.2",                "results/gpt5.2_results.xlsx",                                         _COLORS["GPT-5.2"]),
    "GPT-5-mini":             ("GPT-5-mini",             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",            _COLORS["GPT-5-mini"]),
    "Gemini-3-flash":         ("Gemini-3-flash",         "results/gemini-3-flash-preview_all_personas_results.xlsx",           _COLORS["Gemini-3-flash"]),
    "Claude Opus 4.6":        ("Claude Opus 4.6",        "results/claude-opus-4-6_all_personas_results.xlsx",                  _COLORS["Claude Opus 4.6"]),
    "Qwen3-32B":              ("Qwen3-32B",              "results/Qwen3_32B_all_personas_results.xlsx",                        _COLORS["Qwen3-32B"]),
    "Llama-3.3-70B-Instruct": ("Llama-3.3-70B-Instruct","results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",           _COLORS["Llama-3.3-70B-Instruct"]),
}

LIKERT_VALUES = [-2, -1, 0, 1, 2]
# Shortform labels for x-axis
LIKERT_LABELS = ["SD", "D", "N", "A", "SA"]  # SD=Strongly Disagree, D=Disagree, N=Neutral, A=Agree, SA=Strongly Agree


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_human():
    df = load_human_normalized_data()
    return pd.to_numeric(df["new_belief"], errors="coerce").dropna().astype(int)


def load_model(path):
    df = load_normalized_data(path)
    return pd.to_numeric(df["new_belief"], errors="coerce").dropna().astype(int)


def draw_human_step(ax, series):
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    edges  = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(counts, edges, color=HUMAN_COLOR, lw=1.8, zorder=5, label="Human Post-Stance")


def draw_bars(ax, series, color, title, show_ylabel=True, show_bar_labels=True):
    total  = len(series)
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    x      = np.arange(len(LIKERT_VALUES))

    bars = ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.4, width=0.7, zorder=2, alpha=0.85)

    # Remove numbers and percentage from the plots
    if show_bar_labels:
        for bar, cnt in zip(bars, counts):
            pct = cnt / total * 100 if total > 0 else 0
            if cnt > 0:
                # Offset that scales with bar height
                label_y = bar.get_height() + (max(counts) * 0.02)
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{cnt}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="normal", linespacing=1.2
                )

    # Increase title size by 25% (from 11 to 13.75)
    ax.set_title(title, fontsize=13.75, fontweight="normal", pad=8, color=color, loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(LIKERT_LABELS, fontsize=11.25)

    # Only show y-label for first column (column 0)
    if show_ylabel:
        ax.set_ylabel("Frequency", fontsize=14)
    else:
        ax.set_ylabel("")
    
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11.25)
    
    # Add light grid for readability
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4, zorder=0)


def save_figure(fig, out_stem):
    """Save figure in multiple formats for Nature Communications including SVG."""
    # SVG format (vector, editable)
    fig.savefig(f"{out_stem}.svg", format='svg', bbox_inches='tight')
    print(f"Saved SVG: {out_stem}.svg")

    # PNG format
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches='tight')
    print(f"Saved PNG: {out_stem}.png")

    # PDF format
    fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {out_stem}.pdf")

    # EPS format
    fig.savefig(f"{out_stem}.eps", dpi=300, bbox_inches='tight')
    print(f"Saved EPS: {out_stem}.eps")


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    print("Loading data...")
    model_keys = list(MODELS.keys())

    human = load_human()
    print(f"Human data: n={len(human)}")

    cache = {}
    for key in model_keys:
        _, path, _ = MODELS[key]
        cache[key] = load_model(path)
        print(f"{key}: n={len(cache[key])}")

    y_max = 400

    # FULL PAGE layout: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    
    # Adjust spacing for 2Ã—3 layout
    plt.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.12, 
                        hspace=0.35, wspace=0.25)
    
    model_order = [
        "GPT-5.2", "GPT-5-mini", "Gemini-3-flash",
        "Claude Opus 4.6", "Qwen3-32B", "Llama-3.3-70B-Instruct"
    ]
    
    for idx, key in enumerate(model_order):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        label, _, color = MODELS[key]
        
        # Determine if y-label should be shown (only for column 0)
        show_ylabel = (col == 0)
        
        # Draw bars without numbers and percentages
        draw_bars(ax, cache[key], color, label, show_ylabel=show_ylabel, show_bar_labels=False)
        draw_human_step(ax, human)
        ax.set_ylim(0, y_max)
    
    # Get legend handles from first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Add legend BELOW all subplots, centered
    fig.legend(handles, labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02),
               ncol=2, 
               fontsize=11.25,
               frameon=True,
               framealpha=0.95,
               edgecolor='black',
               fancybox=False,
               borderaxespad=0.5)
    
    # No main title for Nature Communications

    out = "figures/plots/distributions/post_stance_models_2x3"
    print("\nSaving main figure...")
    save_figure(fig, out)

    print(f"\nFigure dimensions: {fig.get_size_inches()} inches")
    print("Layout: 2 rows Ã— 3 columns")
    print("Legend positioned at bottom center")
    print("Frequency label removed from columns 1 and 2")
    print("Numbers and percentages removed from all bars")
    print("Title size increased by 25%")
    
    # Save individual panels as SVG as well
    save_individual_panels(cache, human, model_order)


def save_individual_panels(cache, human, model_order):
    """Save each model panel as a standalone SVG/PNG/PDF for the combined figure."""
    y_max = 400
    slugs = ["gpt52", "gpt5mini", "gemini", "claude", "qwen", "llama"]
    # Match the combined-figure layout: only column-0 models (GPT-5.2 + Claude) keep "Frequency".
    # d, e, g, h (GPT-5-mini, Gemini, Qwen, Llama) have it suppressed at source.
    show_ylabel_by_slug = {"gpt52": True, "gpt5mini": False, "gemini": False,
                           "claude": True, "qwen": False, "llama": False}

    print("\nSaving individual panels...")

    for key, slug in zip(model_order, slugs):
        label, _, color = MODELS[key]
        # Smaller figsize keeps text at the same pt size but shrinks the plot area,
        # so text appears proportionally larger when embedded in the combined figure.
        fig, ax = plt.subplots(figsize=(3.2, 2.2), constrained_layout=True)

        show_ylabel = show_ylabel_by_slug[slug]
        draw_bars(ax, cache[key], color, label, show_ylabel=show_ylabel, show_bar_labels=False)
        draw_human_step(ax, human)
        ax.set_ylim(0, y_max)

        path = f"figures/plots/distributions/panel_{slug}"
        
        # Save as SVG
        fig.savefig(f"{path}.svg", format='svg', bbox_inches='tight')
        print(f"  Saved panel_{slug}.svg")

        # Save as PNG
        fig.savefig(f"{path}.png", dpi=300, bbox_inches='tight')
        print(f"  Saved panel_{slug}.png")

        # Save as PDF
        fig.savefig(f"{path}.pdf", dpi=300, bbox_inches='tight')
        print(f"  Saved panel_{slug}.pdf")
        
        plt.close(fig)


if __name__ == "__main__":
    main()