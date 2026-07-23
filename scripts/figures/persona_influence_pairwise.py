"""
Plot 1: Pairwise output difference â€” base vs no-persona.
For each (persona_id, topic): llm_new_belief_base âˆ’ llm_new_belief_no_persona.
If persona info influences outputs, this distribution should be wide.
If ignored, it should spike at 0.

3 panels (one per model), bar chart of difference frequencies (âˆ’4â€¦+4).
Annotates the % of (persona, topic) pairs that are unchanged (diff = 0).

Run from: HumanSimulationProject/
Output:   claude/persona_influence_pairwise.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


MODELS = {
    "GPT-5-mini": {
        "base":      "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
        "no-persona":"results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx",
    },
    "Qwen3-32B": {
        "base":      "results/Qwen3_32B_all_personas_results.xlsx",
        "no-persona":"results/ablations/Qwen3-32B_no-persona.xlsx",
    },
    "Llama-3.3-70B": {
        "base":      "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "no-persona":"results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx",
    },
}

DIFF_VALS = np.arange(-4, 5)


def normalize(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    return df


def load(path):
    df = pd.read_excel(path)
    return normalize(df)[["persona_id", "topic", "llm_new_belief"]].dropna()


def plot():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(
        "Individual-Level Output Difference: Base vs No-Persona\n"
        r"$\Delta_{\mathrm{diff}} = \mathrm{belief}_{\mathrm{base}} - \mathrm{belief}_{\mathrm{no\text{-}persona}}$"
        "  per (persona, topic) pair",
        fontsize=11, fontweight="bold", y=1.03,
    )

    for ax, (model, paths) in zip(axes, MODELS.items()):
        base = load(paths["base"]).rename(columns={"llm_new_belief": "b_base"})
        nop  = load(paths["no-persona"]).rename(columns={"llm_new_belief": "b_nop"})
        merged = base.merge(nop, on=["persona_id", "topic"], how="inner")
        merged["diff"] = merged["b_base"] - merged["b_nop"]

        counts = merged["diff"].value_counts().reindex(DIFF_VALS, fill_value=0)
        props  = counts / counts.sum()
        pct_zero = props[0] * 100

        bar_colors = ["#bdc3c7" if v == 0 else "#2c3e50" for v in DIFF_VALS]
        bars = ax.bar(DIFF_VALS, props, color=bar_colors, edgecolor="white", width=0.75)

        ax.axvline(0, color="grey", lw=0.8, linestyle=":")
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xlabel("base âˆ’ no-persona  (Likert units)", fontsize=9)
        ax.set_xticks(DIFF_VALS)
        ax.spines[["top", "right"]].set_visible(False)

        # annotate % unchanged
        ax.text(0.97, 0.95, f"{pct_zero:.1f}% unchanged",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdc3c7"))

    axes[0].set_ylabel("Proportion of (persona, topic) pairs", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/plots/persona_influence_pairwise.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/persona_influence_pairwise.pdf", bbox_inches="tight")
    print("Saved figures/plots/persona_influence_pairwise.png")

    # print summary stats
    print(f"\n{'Model':<18}  {'% unchanged':>11}  {'mean |diff|':>11}  {'SD diff':>8}")
    for model, paths in MODELS.items():
        base   = load(paths["base"]).rename(columns={"llm_new_belief": "b_base"})
        nop    = load(paths["no-persona"]).rename(columns={"llm_new_belief": "b_nop"})
        merged = base.merge(nop, on=["persona_id", "topic"], how="inner")
        d      = merged["b_base"] - merged["b_nop"]
        print(f"{model:<18}  {(d==0).mean()*100:>10.1f}%  {d.abs().mean():>11.3f}  {d.std():>8.3f}")


plot()
