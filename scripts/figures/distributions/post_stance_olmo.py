"""
Post-stance distributions for OLMo 32B models at each post-training checkpoint,
with permutation tests against the human baseline.

Layout: 3 rows (SFT â†’ DPO â†’ Final/RLVR) Ã— 2 columns (Think 32B | Instruct 32B).

Run from: HumanSimulationProject/
    python claude/scripts/distributions/post_stance_olmo.py [--permutation-test]
Output:   figures/plots/distributions/post_stance_olmo.png / .pdf
"""

import sys

import argparse
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yaml
from scipy.stats import permutation_test as scipy_permutation_test

from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


# â”€â”€ custom loader for Think models (new_belief lives in raw_response JSON) â”€â”€â”€â”€



def _extract_new_belief_from_raw(raw_response):
    """Parse new_belief integer out of the JSON block in a Think model response."""
    if not isinstance(raw_response, str):
        return None
    # The JSON block follows </think> for think-format responses
    m = re.search(r'</think>\s*(\{.*\})', raw_response, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            val = obj.get("new_belief")
            if val is not None:
                return int(val)
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: plain regex anywhere in the response
    m = re.search(r'"new_belief"\s*:\s*(-?\d+)', raw_response)
    if m:
        return int(m.group(1))
    return None


def load_think_model(path):
    """Load a Think-format results file, extracting new_belief from raw_response."""
    df = pd.read_excel(path)
    df["new_belief"] = pd.to_numeric(
        df["raw_response"].apply(_extract_new_belief_from_raw),
        errors="coerce",
    )
    # Apply the same sign-flip normalisation as load_normalized_data
    if "statement_formulation" in df.columns:
        neg_mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
        df.loc[neg_mask, "new_belief"] *= -1
    # Rename to match the shared schema
    df = df.rename(columns={"init_belief": "init_stance"})
    n_nan = df["new_belief"].isna().sum()
    if n_nan > 0:
        print(f"  NOTE: {n_nan}/{len(df)} rows have unparseable new_belief in {path}")
    return df

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
HUMAN_COLOR = _yaml["human"]

# Two model families, each with 3 training-stage checkpoints.
# Rows = stages, columns = model families.
# "think": True  â†’ new_belief must be extracted from raw_response JSON.
MODELS = [
    {
        "name": "Olmo-3-32B-Think",
        "color": "#7fc97f",           # muted green â€” not in colors.yaml
        "think": True,
        "variants": [
            ("SFT",   "results/Olmo-3-32B-Think-SFT_all_personas_results.xlsx"),
            ("DPO",   "results/Olmo-3-32B-Think-DPO_all_personas_results.xlsx"),
            ("Final", "results/Olmo-3-32B-Think_all_personas_results.xlsx"),
        ],
    },
    {
        "name": "Olmo-3.1-32B-Instruct",
        "color": "#e78ac3",           # muted pink â€” not in colors.yaml
        "think": False,
        "variants": [
            ("SFT",   "results/Olmo-3.1-32B-Instruct-SFT_all_personas_results.xlsx"),
            ("DPO",   "results/Olmo-3.1-32B-Instruct-DPO_all_personas_results.xlsx"),
            ("Final", "results/Olmo-3.1-32B-Instruct_all_personas_results.xlsx"),
        ],
    },
]

LIKERT_VALUES = [-2, -1, 0, 1, 2]
LIKERT_LABELS = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]


# â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_human():
    df = load_human_normalized_data()
    series = pd.to_numeric(df["new_belief"], errors="coerce").dropna().astype(int)
    return df, series


def load_model(path, think=False):
    df = load_think_model(path) if think else load_normalized_data(path)
    series = pd.to_numeric(df["new_belief"], errors="coerce").dropna().astype(int)
    return df, series


def perform_permutation_test(sample_1, sample_2):
    def statistic(x, y):
        return np.mean(x - y)

    N_TOTAL = 1_000_000
    CHUNK_SIZE = 100_000
    N_CHUNKS = N_TOTAL // CHUNK_SIZE
    count = 0
    obs = statistic(sample_1, sample_2)
    rng = np.random.default_rng(42)
    for _ in range(N_CHUNKS):
        res = scipy_permutation_test(
            (sample_1, sample_2),
            statistic,
            permutation_type="samples",
            random_state=rng,
            n_resamples=CHUNK_SIZE,
        )
        count += np.sum(np.abs(res.null_distribution) >= np.abs(obs))
    p_value = count / N_TOTAL
    alpha = 0.0167  # Bonferroni-corrected for 3 comparisons per family
    return obs, p_value, p_value < alpha


def run_permutation_test(model_df, human_df, label):
    merged = pd.merge(
        model_df[["persona_id", "topic", "new_belief"]],
        human_df[["persona_id", "topic", "new_belief"]],
        on=["persona_id", "topic"],
        how="inner",
        suffixes=("_model", "_human"),
    )
    n_before = len(merged)
    merged = merged.dropna(subset=["new_belief_model", "new_belief_human"])
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        print(f"  WARNING: dropped {n_dropped}/{n_before} rows with NaN new_belief for '{label}'")
    sample_model = pd.to_numeric(merged["new_belief_model"]).to_numpy()
    sample_human = pd.to_numeric(merged["new_belief_human"]).to_numpy()
    obs_diff, p_value, reject = perform_permutation_test(sample_model, sample_human)
    sig = "***" if reject else "n.s."
    print(f"  {label:50s}  obs_diff={obs_diff:+.4f}  p={p_value:.4f}  reject H0={reject}  ({sig})")
    return obs_diff, p_value, reject


def draw_human_step(ax, series):
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    edges = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(counts, edges, color=HUMAN_COLOR, lw=1.8, zorder=5, label="Human Post-Stance")


def draw_bars(ax, series, color, title, p_value=None, reject=None):
    total = len(series)
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    x = np.arange(len(LIKERT_VALUES))

    bars = ax.bar(x, counts, color=color, edgecolor="white", linewidth=0.4, width=0.7)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3,
            f"{cnt}\n({pct:.0f}%)",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(LIKERT_LABELS, fontsize=7)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)

    if p_value is not None:
        sig_label = f"p={p_value:.4f} {'***' if reject else 'n.s.'}"
        ax.text(
            0.98, 0.97, sig_label,
            transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            color="black" if reject else "#999999",
            fontweight="bold" if reject else "normal",
        )


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--permutation-test", action="store_true",
        help="Run permutation tests against the human baseline (slow)",
    )
    args = parser.parse_args()

    human_df, human_series = load_human()

    # Load all data: all_data[col][row] = (df, series)
    # col = model family index, row = stage index
    all_data = []
    for model_cfg in MODELS:
        col_data = []
        for _stage_label, path in model_cfg["variants"]:
            df, series = load_model(path, think=model_cfg["think"])
            col_data.append((df, series))
        all_data.append(col_data)

    # Permutation tests â€” keyed by (model_idx, stage_idx)
    perm_results = {}
    if args.permutation_test:
        print("\n--- Permutation Tests vs. Human Baseline (alpha=0.0167, Bonferroni-corrected) ---")
        for model_idx, (model_cfg, col_data) in enumerate(zip(MODELS, all_data)):
            for stage_idx, ((stage_label, _), (df, _)) in enumerate(zip(model_cfg["variants"], col_data)):
                label = f"{model_cfg['name']} ({stage_label})"
                obs_diff, p_value, reject = run_permutation_test(df, human_df, label)
                perm_results[(model_idx, stage_idx)] = (obs_diff, p_value, reject)

    # Global y-max
    all_series = [s for col_data in all_data for (_, s) in col_data] + [human_series]
    global_max = max(
        max(int((s == v).sum()) for v in LIKERT_VALUES)
        for s in all_series
    )
    y_max = global_max + 20

    # Layout: 2 rows (model families) Ã— 3 columns (stages)
    n_families = len(MODELS)                # 2
    n_stages   = len(MODELS[0]["variants"]) # 3
    fig, axes = plt.subplots(n_families, n_stages, figsize=(12, 8), constrained_layout=True)

    for row_idx, model_cfg in enumerate(MODELS):
        color = model_cfg["color"]
        for col_idx, (stage_label, _) in enumerate(model_cfg["variants"]):
            ax = axes[row_idx, col_idx]
            df, series = all_data[row_idx][col_idx]
            title = f"{model_cfg['name']}\n({stage_label})"
            p_val = perm_results.get((row_idx, col_idx))
            if p_val is not None:
                draw_bars(ax, series, color, title, p_value=p_val[1], reject=p_val[2])
            else:
                draw_bars(ax, series, color, title)
            draw_human_step(ax, human_series)
            ax.set_ylim(0, y_max)

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg,
        loc="lower center", ncol=1,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    out = "figures/plots/distributions/post_stance_olmo"
    fig.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved {out}.svg / .png / .pdf")


main()
