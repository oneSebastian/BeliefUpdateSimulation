"""
Belief change (Î” = post âˆ’ initial) distribution by topic, comparing models.
3 rows (topics: UBI, penalty, weight_loss) Ã— 6 columns (one per model).
Each panel overlays human bars + a single LLM dot-line.

Run from: HumanSimulationProject/
Output:   figures/plots/delta_belief_by_topic_models.png / .pdf
"""

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]

# â”€â”€ model â†’ (display label, data file, color) â€” from colors.yaml â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

HUMAN_COLOR = _yaml["human"]
TOPICS      = ["UBI", "penalty", "weight_loss"]
DELTA_BINS  = np.arange(-4.5, 5.5, 1.0)
DELTA_TICKS = np.arange(-4, 5)


# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    return df[["topic", "delta"]].dropna()


def proportions(delta: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(delta, bins=DELTA_BINS)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(float)


# â”€â”€ panel drawing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_panel(ax, human_delta, llm_delta, color, show_ylabel, show_xlabel):
    # LLM: coloured bars (foreground)
    l_props = proportions(llm_delta)
    ax.bar(DELTA_TICKS, l_props, width=0.7, color=color, edgecolor="white",
           linewidth=0.4, zorder=2)

    # Human: grey step outline (background reference)
    h_props = proportions(human_delta)
    edges = np.append(DELTA_TICKS - 0.5, DELTA_TICKS[-1] + 0.5)
    ax.stairs(h_props, edges, color=HUMAN_COLOR, lw=1.8, zorder=3, label="Human")

    ax.axvline(0, color="grey", lw=0.7, linestyle=":", zorder=0)
    ax.set_xticks(DELTA_TICKS)
    ax.set_xticklabels([str(v) for v in DELTA_TICKS], fontsize=6)
    ax.set_xlim(-4.7, 4.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)

    if show_ylabel:
        ax.set_ylabel("Proportion", fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Î” Belief  (post âˆ’ initial)", fontsize=8)


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot():
    human_df = load_human()

    model_keys  = list(MODELS.keys())
    n_models    = len(model_keys)

    # pre-load all LLM data
    llm_cache = {}
    for key in model_keys:
        _, path, _ = MODELS[key]
        llm_cache[key] = load_llm(path)

    fig, axes = plt.subplots(
        3, n_models,
        figsize=(3.2 * n_models, 10),
        sharex=True, sharey=False,
    )


    for row, topic in enumerate(TOPICS):
        for col, key in enumerate(model_keys):
            ax             = axes[row, col]
            label, _, color = MODELS[key]

            human_delta = human_df.loc[human_df["topic"] == topic, "delta"].values
            llm_delta   = llm_cache[key].loc[llm_cache[key]["topic"] == topic, "delta"].values

            draw_panel(
                ax,
                human_delta=human_delta,
                llm_delta=llm_delta,
                color=color,
                show_ylabel=(col == 0),
                show_xlabel=(row == 2),
            )

            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", pad=6, color=color)

            if col == 0:
                ax.set_ylabel(f"{TOPIC_LABELS[topic]}\n\nProportion", fontsize=9)

    # shared legend: grey line for Human, coloured patch per model
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    legend_handles = [
        Line2D([0], [0], color=HUMAN_COLOR, lw=1.8, label="Human"),
    ] + [
        mpatches.Patch(color=MODELS[k][2], label=MODELS[k][0])
        for k in model_keys
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=n_models + 1,
        fontsize=8, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    out = "figures/plots/delta_belief_by_topic_models"
    plt.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    plt.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved {out}.svg / .png / .pdf")


plot()
