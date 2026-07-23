"""
Plot 1: Per-persona consistency â€” Human vs all LLMs (base condition).

Run from: HumanSimulationProject/
Output:   claude/consistency_across_llms.png
"""

import yaml
import matplotlib.pyplot as plt
import sys, os
from scripts.figures.consistency_utils import load_condition, draw_icc_and_violin

with open("colors.yaml") as _f:
    _COLORS = yaml.safe_load(_f)["models"]

ENTRIES = [
    ("Human",                 "#757575",                          "human"),
    ("GPT-5.2",               _COLORS["GPT-5.2"],                "results/gpt5.2_results.xlsx"),
    ("GPT-5-mini",            _COLORS["GPT-5-mini"],             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx"),
    ("Claude Opus 4.6",       _COLORS["Claude Opus 4.6"],        "results/claude-opus-4-6_all_personas_results.xlsx"),
    ("Gemini-3-flash",        _COLORS["Gemini-3-flash"],         "results/gemini-3-flash-preview_all_personas_results.xlsx"),
    ("Qwen3-32B",             _COLORS["Qwen3-32B"],              "results/Qwen3_32B_all_personas_results.xlsx"),
    ("Llama-3.3-70B-Instruct",_COLORS["Llama-3.3-70B-Instruct"],"results/Llama-3.3-70B-Instruct_all_personas_results.xlsx"),
]


def plot():
    entries = [(label, color, load_condition(spec))
               for label, color, spec in ENTRIES]

    fig, (ax_icc, ax_violin) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Per-Persona Consistency Across Topics â€” Human vs LLMs (base condition)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    draw_icc_and_violin(ax_icc, ax_violin, entries)

    plt.tight_layout()
    plt.savefig("figures/plots/consistency_across_llms.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/consistency_across_llms.pdf", bbox_inches="tight")
    print("Saved figures/plots/consistency_across_llms.png")

    print("\nICC (1,1):")
    import numpy as np
    from scripts.figures.consistency_utils import icc_one_way
    for label, _, df in entries:
        print(f"  {label:<20s}  {icc_one_way(df):.4f}")


plot()
