"""
Human ground truth: Post-Stance Distribution as bars, Initial Stance as outline.
Single panel, styled to match post_stance_models.pdf.

Run from: HumanSimulationProject/
Output:   figures/plots/distributions/human_distributions.svg / .png / .pdf / .eps
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from belief_update_sim.data_loading import load_human_normalized_data
import os

# Set Nature Communications style parameters
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

POST_COLOR = "#757575"   # dark grey â€” post stance (matches HUMAN_COLOR in other plots)
INIT_COLOR = "#cccccc"   # light grey â€” initial stance (outline)

LIKERT_VALUES = [-2, -1, 0, 1, 2]
# Shortform labels matching the model distribution plot
LIKERT_LABELS = ["SD", "D", "N", "A", "SA"]  # SD=Strongly Disagree, D=Disagree, N=Neutral, A=Agree, SA=Strongly Agree


def draw_bars(ax, series, color, label):
    s      = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    counts = [int((s == v).sum()) for v in LIKERT_VALUES]
    x      = np.arange(len(LIKERT_VALUES))

    ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.4, width=0.7,
           label="Human", alpha=0.85, zorder=2)

    return counts


def draw_init_step(ax, series, color, label):
    s      = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    counts = [int((s == v).sum()) for v in LIKERT_VALUES]
    edges  = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(counts, edges, color=color, lw=2.0, zorder=5, label=label)


def save_figure(fig, out_stem):
    """Save figure in multiple formats including SVG."""
    # SVG format (vector, editable)
    fig.savefig(f"{out_stem}.svg", format='svg', bbox_inches='tight')
    print(f"Saved SVG: {out_stem}.svg")

    # PNG format (raster, for preview)
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches='tight')
    print(f"Saved PNG: {out_stem}.png")

    # PDF format
    fig.savefig(f"{out_stem}.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved PDF: {out_stem}.pdf")

    # EPS format
    fig.savefig(f"{out_stem}.eps", dpi=300, bbox_inches='tight')
    print(f"Saved EPS: {out_stem}.eps")


def main():
    print("Loading human data...")
    df   = load_human_normalized_data()
    init = df["init_stance"]
    post = df["new_belief"]

    post_s = pd.to_numeric(post, errors="coerce").dropna().astype(int)
    init_s = pd.to_numeric(init, errors="coerce").dropna().astype(int)
    
    print(f"Post-stance n={len(post_s)}")
    print(f"Initial stance n={len(init_s)}")

    y_max = 400

    # Create figure - single panel sized for embedding in the combined figure.
    # Smaller figsize keeps text at the same pt size but shrinks the plot area,
    # so text appears proportionally larger when embedded.
    fig, ax = plt.subplots(figsize=(3.2, 2.2))

    draw_bars(ax, post, POST_COLOR, "Post-Stance")
    draw_init_step(ax, init, INIT_COLOR, "Initial Stance")

    # Styling to match model distribution plots with consistent fonts
    ax.set_title("Human Ground Truth", fontsize=13.75, fontweight="normal", pad=8,
                 color=POST_COLOR, loc='left')
    ax.set_xticks(np.arange(len(LIKERT_VALUES)))
    ax.set_xticklabels(LIKERT_LABELS, fontsize=11.25)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.set_ylim(0, y_max)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11.25)  # Matching model plot
    
    # Add light grid for readability
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4, zorder=0)

    # Save clean version (no legend) for use in combined figure
    print("\nSaving version without legend...")
    for ext in ['.svg', '.png', '.pdf']:
        fig.savefig(f"figures/plots/distributions/human_distributions_nol{ext}",
                    format=ext.replace('.', '') if ext != '.png' else None,
                    dpi=300 if ext == '.png' else None,
                    bbox_inches='tight')
        print(f"Saved: figures/plots/distributions/human_distributions_nol{ext}")

    # Add legend once for standalone version with consistent font size
    fig.legend(*ax.get_legend_handles_labels(),
               loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=2, fontsize=11.25, frameon=True, framealpha=0.95,  # Increased from 9 to 11.25
               edgecolor='black', fancybox=False)

    out = "figures/plots/distributions/human_distributions"
    print("\nSaving standalone version with legend...")
    save_figure(fig, out)
    
    print(f"\nFigure dimensions: {fig.get_size_inches()} inches")
    print("\nHuman ground truth distribution saved successfully!")
    print(f"   Location: figures/plots/distributions/")
    print(f"   Files created:")
    print(f"     - human_distributions.svg (vector, editable)")
    print(f"     - human_distributions.png")
    print(f"     - human_distributions.pdf")
    print(f"     - human_distributions.eps")
    print(f"   Also saved (no legend):")
    print(f"     - human_distributions_nol.svg")
    print(f"     - human_distributions_nol.png")
    print(f"     - human_distributions_nol.pdf")
    print("\nFont sizes updated to match model distribution plots:")
    print("   - X-axis labels: 11.25 (was 9)")
    print("   - Y-axis label: 12.5 (was 10)")
    print("   - Tick labels: 11.25 (was 9)")
    print("   - Legend: 11.25 (was 9)")
    print("   - Title: 13.75 (was 11)")
    print("X-axis labels changed to shortform: SD, D, N, A, SA")


if __name__ == "__main__":
    main()