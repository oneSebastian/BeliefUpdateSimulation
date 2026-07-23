"""
Plot 2: Per-persona consistency â€” Human vs ablation conditions
        for GPT-5-mini, Qwen3-32B, and Llama-3.3-70B.

Entries grouped by model: Human | GPT Ã— 4 | Qwen Ã— 4 | Llama Ã— 4.
Vertical separator lines mark model boundaries.

Run from: HumanSimulationProject/
Output:   claude/consistency_ablations.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import sys, os
from scripts.figures.consistency_utils import load_condition, draw_icc_and_violin, icc_one_way, within_persona_sd

# â”€â”€ condition specs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each model block: (display_label, hex_color, file_path)
MODEL_BLOCKS = {
    "GPT-5-mini": [
        ("Base",           "#e74c3c", "results/academic-ai-gpt-5-mini_all_personas_results.xlsx"),
        ("No demographic", "#f1948a", "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx"),
        ("No personality", "#fadbd8", "results/ablations/academic-ai-gpt-5-mini_no-personality.xlsx"),
        ("No persona",     "#922b21", "results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx"),
    ],
    "Qwen3-32B": [
        ("Base",           "#e67e22", "results/Qwen3_32B_all_personas_results.xlsx"),
        ("No demographic", "#f0b27a", "results/ablations/Qwen3-32B_no-demographic.xlsx"),
        ("No personality", "#fdebd0", "results/ablations/Qwen3-32B_no-personality.xlsx"),
        ("No persona",     "#935116", "results/ablations/Qwen3-32B_no-persona.xlsx"),
    ],
    "Llama-3.3-70B": [
        ("Base",           "#8e44ad", "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx"),
        ("No demographic", "#c39bd3", "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx"),
        ("No personality", "#ebdef0", "results/ablations/Llama-3.3-70B-Instruct_no-personality.xlsx"),
        ("No persona",     "#6c3483", "results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx"),
    ],
}

with open("colors.yaml") as _f:
    HUMAN_COLOR = yaml.safe_load(_f)["human"]


def plot():
    # build flat entry list: (label, color, df)
    human_df = load_condition("human")
    entries  = [("Human", HUMAN_COLOR, human_df)]

    # separator positions: after human (pos 0) and after each model block
    separator_positions = []   # x positions for vertical lines

    pos = 1
    block_midpoints = {}   # model name â†’ midpoint x for annotation
    for model, conditions in MODEL_BLOCKS.items():
        block_midpoints[model] = pos + (len(conditions) - 1) / 2
        for label, color, spec in conditions:
            entries.append((label, color, load_condition(spec)))
            pos += 1
        separator_positions.append(pos - 0.5)

    separator_positions = separator_positions[:-1]  # no separator after last block

    # compute stats
    icc_vals = [icc_one_way(df) for _, _, df in entries]
    sd_vals  = [within_persona_sd(df) for _, _, df in entries]
    labels   = [e[0] for e in entries]
    colors   = [e[1] for e in entries]

    fig, (ax_icc, ax_violin) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(
        "Per-Persona Consistency â€” Human vs Ablation Conditions "
        "(GPT-5-mini Â· Qwen3-32B Â· Llama-3.3-70B)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for ax, vals, ylabel, title in [
        (ax_icc,    icc_vals, "ICC (1,1)",             "Intraclass Correlation"),
        (ax_violin, sd_vals,  "Within-persona SD of Î”", "Within-Persona Spread\n(lower = more consistent)"),
    ]:
        if ax is ax_icc:
            bars = ax.bar(range(len(labels)), vals, color=colors,
                          edgecolor="white", width=0.6, zorder=2)
            ax.axhline(0, color="grey", lw=0.8, linestyle=":")
            for bar, val in zip(bars, vals):
                yoff = 0.004 if val >= 0 else -0.012
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + yoff, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
            ymin = min(0, min(vals)) - 0.04
            ymax = max(0, max(vals)) + 0.08
            ax.set_ylim(ymin, ymax)
        else:
            vparts = ax.violinplot(vals, positions=range(len(labels)),
                                   showmedians=True, showextrema=False)
            for pc, col in zip(vparts["bodies"], colors):
                pc.set_facecolor(col)
                pc.set_alpha(0.6)
            vparts["cmedians"].set_color("black")
            vparts["cmedians"].set_linewidth(1.5)
            for i, (v, col) in enumerate(zip(vals, colors)):
                ax.scatter(i, np.mean(v), color=col,
                           edgecolors="white", s=35, zorder=5)

        # model block separators
        ylo, yhi = ax.get_ylim()
        for xpos in separator_positions:
            ax.axvline(xpos, color="grey", lw=1.2, linestyle="--", alpha=0.6, zorder=0)

        # model name annotations just above x-axis
        for model, mid in block_midpoints.items():
            ax.text(mid, ylo + (yhi - ylo) * 0.02, model,
                    ha="center", va="bottom", fontsize=8.5,
                    fontstyle="italic", color="dimgrey")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("figures/plots/consistency_ablations.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/consistency_ablations.pdf", bbox_inches="tight")
    print("Saved figures/plots/consistency_ablations.png")

    print("\nICC (1,1):")
    for label, _, df in entries:
        print(f"  {label:<20s}  {icc_one_way(df):.4f}")


plot()
