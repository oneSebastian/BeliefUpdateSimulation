"""Shared helpers for persona-consistency plots."""

import numpy as np
import pandas as pd
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS



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
    df = pd.read_excel(spec)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df[["persona_id", "topic", "delta"]].dropna()


def icc_one_way(df: pd.DataFrame, k: int = 3) -> float:
    """One-way random-effects ICC(1,1) for a balanced design."""
    groups      = df.groupby("persona_id")["delta"]
    n           = df["persona_id"].nunique()
    grand_mean  = df["delta"].mean()
    group_means = groups.mean()
    group_sizes = groups.count()

    ss_between  = ((group_means - grand_mean) ** 2 * group_sizes).sum()
    ss_within   = groups.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()

    ms_between  = ss_between / (n - 1)
    ms_within   = ss_within  / (n * (k - 1))

    return float((ms_between - ms_within) / (ms_between + (k - 1) * ms_within))


def within_persona_sd(df: pd.DataFrame) -> np.ndarray:
    """SD of Δ across topics per persona_id."""
    return df.groupby("persona_id")["delta"].std(ddof=1).dropna().values


def draw_icc_and_violin(ax_icc, ax_violin, entries, title_suffix=""):
    """
    entries: list of (label, color, delta_df)
    Draws ICC bar chart on ax_icc and within-persona SD violin on ax_violin.
    """
    import matplotlib.pyplot as plt

    labels = [e[0] for e in entries]
    colors = [e[1] for e in entries]
    dfs    = [e[2] for e in entries]

    icc_vals = [icc_one_way(df) for df in dfs]
    sd_vals  = [within_persona_sd(df) for df in dfs]

    # ── ICC bar ──────────────────────────────────────────────────────────────
    bars = ax_icc.bar(range(len(labels)), icc_vals, color=colors,
                      edgecolor="white", width=0.6, zorder=2)
    ax_icc.axhline(0, color="grey", lw=0.8, linestyle=":")
    for bar, val in zip(bars, icc_vals):
        yoff = 0.004 if val >= 0 else -0.012
        ax_icc.text(bar.get_x() + bar.get_width() / 2,
                    val + yoff, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax_icc.set_xticks(range(len(labels)))
    ax_icc.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax_icc.set_ylabel("ICC (1,1)", fontsize=10)
    ax_icc.set_title(f"Intraclass Correlation{title_suffix}", fontsize=10,
                     fontweight="bold")
    ax_icc.spines[["top", "right"]].set_visible(False)
    ymin = min(0, min(icc_vals)) - 0.04
    ymax = max(0, max(icc_vals)) + 0.06
    ax_icc.set_ylim(ymin, ymax)

    # ── violin ───────────────────────────────────────────────────────────────
    vparts = ax_violin.violinplot(sd_vals, positions=range(len(labels)),
                                  showmedians=True, showextrema=False)
    for pc, col in zip(vparts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    vparts["cmedians"].set_color("black")
    vparts["cmedians"].set_linewidth(1.5)
    for i, (vals, col) in enumerate(zip(sd_vals, colors)):
        ax_violin.scatter(i, np.mean(vals), color=col,
                          edgecolors="white", s=40, zorder=5)
    ax_violin.set_xticks(range(len(labels)))
    ax_violin.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax_violin.set_ylabel("Within-persona SD of Δ", fontsize=10)
    ax_violin.set_title(f"Within-Persona Spread{title_suffix}\n(lower = more consistent)",
                        fontsize=10, fontweight="bold")
    ax_violin.spines[["top", "right"]].set_visible(False)
