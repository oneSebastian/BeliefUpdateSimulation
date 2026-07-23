"""
Per-persona consistency across topics.

If LLMs use persona information, the same simulated person should respond
consistently across topics (high ICC, low within-persona SD).
Without persona, responses should be more topic-driven and inconsistent.

Three panels:
  1. ICC(1,1) per condition â€” bar chart
  2. Within-persona SD distribution â€” violin plot
  3. Persona Ã— topic Î” heatmaps â€” one per key condition (top 40 most variable personas)

Run from: HumanSimulationProject/
Output:   claude/persona_consistency.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


# Ordered for display: human reference, then conditions from most to least persona info
CONDITIONS = {
    "Human":             "human",
    "Base":              "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
    "No demographic":    "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx",
    "No personality":    "results/ablations/academic-ai-gpt-5-mini_no-personality.xlsx",
    "No persona":        "results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx",
}

# Conditions to show in the heatmap comparison
HEATMAP_CONDITIONS = ["Human", "Base", "No persona"]

TOPICS = ["UBI", "penalty", "weight_loss"]
TOPIC_LABELS = {"UBI": "UBI", "penalty": "Penalty", "weight_loss": "Wt. Loss"}

# Colour per condition
COND_COLORS = {
    "Human":           "#2980b9",
    "Base":            "#1a1a2e",
    "No demographic":  "#8e44ad",
    "No personality":  "#e67e22",
    "No persona":      "#e74c3c",
}

N_HEATMAP_PERSONAS = 40   # show top-N most variable personas in human data

# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_condition(spec: str) -> pd.DataFrame:
    if spec == "human":
        df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
        df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
        return df[["persona_id", "topic", "delta"]]
    else:
        df = pd.read_excel(spec)
        df = normalize(df)
        df["delta"] = df["llm_new_belief"] - df["init_belief"]
        return df[["persona_id", "topic", "delta"]].dropna()


# â”€â”€ ICC (one-way random effects, ICC 1,1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def icc_one_way(df: pd.DataFrame, k: int = 3) -> float:
    """
    One-way random effects ICC for a balanced design.
    k = number of repeated measurements per subject (topics).
    """
    groups  = df.groupby("persona_id")["delta"]
    n       = df["persona_id"].nunique()

    grand_mean  = df["delta"].mean()
    group_means = groups.mean()
    group_sizes = groups.count()

    ss_between  = ((group_means - grand_mean) ** 2 * group_sizes).sum()
    ss_within   = groups.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()

    ms_between  = ss_between / (n - 1)
    ms_within   = ss_within  / (n * (k - 1))

    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    return float(icc)


# â”€â”€ within-persona SD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def within_persona_sd(df: pd.DataFrame) -> np.ndarray:
    """SD of Î” across topics, per persona_id."""
    return df.groupby("persona_id")["delta"].std(ddof=1).dropna().values


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot():
    # load all conditions
    data = {label: load_condition(spec) for label, spec in CONDITIONS.items()}

    icc_vals  = {label: icc_one_way(df) for label, df in data.items()}
    wpsd_vals = {label: within_persona_sd(df) for label, df in data.items()}

    # choose top-N most variable personas from human data for heatmaps
    human_var = (
        data["Human"]
        .groupby("persona_id")["delta"].std()
        .nlargest(N_HEATMAP_PERSONAS)
    )
    top_personas = human_var.index.tolist()

    labels     = list(CONDITIONS.keys())
    icc_list   = [icc_vals[l] for l in labels]
    colors     = [COND_COLORS[l] for l in labels]

    # â”€â”€ layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Per-Persona Consistency Across Topics (GPT-5-mini, all conditions)",
        fontsize=13, fontweight="bold", y=1.005,
    )

    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38,
                              height_ratios=[1, 1.1])

    # top row: ICC bar + violin side by side
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                           wspace=0.35)
    ax_icc    = fig.add_subplot(top[0])
    ax_violin = fig.add_subplot(top[1])

    # bottom row: 4 heatmaps
    bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                           wspace=0.08)
    hm_axes = [fig.add_subplot(bot[i]) for i in range(4)]

    # â”€â”€ Panel 1: ICC bar chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bars = ax_icc.bar(labels, icc_list, color=colors, edgecolor="white",
                      width=0.6, zorder=2)
    ax_icc.axhline(0, color="grey", lw=0.8, linestyle=":")
    for bar, val in zip(bars, icc_list):
        ypos = val + 0.005 if val >= 0 else val - 0.015
        ax_icc.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")
    ax_icc.set_ylabel("ICC (1,1)", fontsize=10)
    ax_icc.set_title("Intraclass Correlation across Topics", fontsize=10,
                     fontweight="bold")
    ax_icc.tick_params(axis="x", rotation=25, labelsize=9)
    ax_icc.spines[["top", "right"]].set_visible(False)
    ax_icc.set_ylim(min(0, min(icc_list)) - 0.04,
                    max(icc_list) + 0.06)

    # â”€â”€ Panel 2: within-persona SD violin â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vdata  = [wpsd_vals[l] for l in labels]
    vparts = ax_violin.violinplot(vdata, positions=range(len(labels)),
                                  showmedians=True, showextrema=False)

    for i, (pc, lbl) in enumerate(zip(vparts["bodies"], labels)):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    vparts["cmedians"].set_color("black")
    vparts["cmedians"].set_linewidth(1.5)

    # overlay mean dots
    for i, vals in enumerate(vdata):
        ax_violin.scatter(i, np.mean(vals), color=colors[i],
                          edgecolors="white", s=40, zorder=5)

    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax_violin.set_ylabel("Within-persona SD of Î”", fontsize=10)
    ax_violin.set_title("Within-Persona Spread Across Topics\n"
                        "(lower = more consistent)", fontsize=10,
                        fontweight="bold")
    ax_violin.spines[["top", "right"]].set_visible(False)

    # â”€â”€ Panel 3: persona Ã— topic heatmaps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    vmin, vmax = -4, 4

    for ax, cond_label in zip(hm_axes, HEATMAP_CONDITIONS):
        df   = data[cond_label]
        sub  = df[df["persona_id"].isin(top_personas)]
        heat = (
            sub.groupby(["persona_id", "topic"])["delta"]
            .mean()
            .unstack("topic")
            .reindex(columns=TOPICS)
            .reindex(top_personas)
        )
        heat.columns = [TOPIC_LABELS[t] for t in TOPICS]

        im = ax.imshow(heat.values, cmap="RdBu", vmin=vmin, vmax=vmax,
                       aspect="auto")

        ax.set_title(cond_label, fontsize=9, fontweight="bold",
                     color=COND_COLORS[cond_label])
        ax.set_xticks(range(3))
        ax.set_xticklabels(heat.columns, fontsize=8)
        ax.set_yticks([])
        ax.set_xlabel("Topic", fontsize=8)

    hm_axes[0].set_ylabel(
        f"Personas (top {N_HEATMAP_PERSONAS} by human Î” variance)",
        fontsize=8.5,
    )

    # shared colorbar for heatmaps
    cbar = fig.colorbar(im, ax=hm_axes, shrink=0.7, pad=0.02,
                        label="Mean Î” Belief")
    cbar.ax.tick_params(labelsize=8)

    plt.savefig("figures/plots/persona_consistency.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/persona_consistency.pdf", bbox_inches="tight")
    print("Saved figures/plots/persona_consistency.png")

    # print ICC table
    print("\nICC (1,1) per condition:")
    for lbl, val in icc_vals.items():
        print(f"  {lbl:<20s}  {val:.4f}")


plot()
