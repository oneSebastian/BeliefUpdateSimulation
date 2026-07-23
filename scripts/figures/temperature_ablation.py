"""
Temperature ablation: post-stance distributions and permutation tests.
Compares GPT-5-mini, Qwen3-32B, and Llama-3.3-70B-Instruct at temperatures
0.0, 0.7, and 2.0 against the human baseline. Layout: 3 rows (models) x 3 columns (temps).

Run from: HumanSimulationProject/
Output:   figures/plots/distributions/post_stance_temperature_ablation.png / .pdf
"""

import sys

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import permutation_test as scipy_permutation_test
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data


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
    alpha = 0.0167
    return obs, p_value, p_value < alpha


with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]
HUMAN_COLOR = _yaml["human"]

# 3 models x 3 temperatures: rows = models, columns = temperatures
MODELS = [
    {
        "name": "GPT-5-Mini",
        "color": _COLORS["GPT-5-mini"],
        "variants": [
            ("T=0.0", "results/ablations/academic-ai-gpt-5-mini_temperature_0.0.xlsx"),
            ("T=0.7", "results/academic-ai-gpt-5-mini_all_personas_results.xlsx"),
            ("T=2.0", "results/ablations/academic-ai-gpt-5-mini_temperature_2.0.xlsx"),
        ],
    },
    {
        "name": "Qwen3-32B",
        "color": _COLORS["Qwen3-32B"],
        "variants": [
            ("T=0.0", "results/ablations/Qwen3_32B_t0.0.xlsx"),
            ("T=0.7", "results/Qwen3_32B_all_personas_results.xlsx"),
            ("T=2.0", "results/ablations/Qwen3_32B_t2.0.xlsx"),
        ],
    },
    {
        "name": "Llama-3.3-70B-Instruct",
        "color": _COLORS["Llama-3.3-70B-Instruct"],
        "variants": [
            ("T=0.0", "results/ablations/Llama-3.3-70B-Instruct_t0.0.xlsx"),
            ("T=0.7", "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx"),
            ("T=2.0", "results/ablations/Llama-3.3-70B-Instruct_t2.0.xlsx"),
        ],
    },
]

LIKERT_VALUES = [-2, -1, 0, 1, 2]
LIKERT_LABELS = ["Strongly\nDisagree", "Disagree", "Neutral", "Agree", "Strongly\nAgree"]


def load_human():
    df = load_human_normalized_data()
    series = pd.to_numeric(df["new_belief"], errors="coerce").dropna().astype(int)
    return df, series


def load_model(path):
    df = load_normalized_data(path)
    series = pd.to_numeric(df["new_belief"], errors="coerce").dropna().astype(int)
    return df, series


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
    print(f"  {label:40s}  obs_diff={obs_diff:+.4f}  p={p_value:.4f}  reject H0={reject}  ({sig})")
    return obs_diff, p_value, reject


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutation-test", action="store_true",
                        help="Run permutation tests against the human baseline")
    args = parser.parse_args()

    human_df, human_series = load_human()

    # Load all data: models[row][col] = (df, series)
    all_data = []
    for model_cfg in MODELS:
        row_data = []
        for temp_label, path in model_cfg["variants"]:
            df, series = load_model(path)
            row_data.append((df, series))
        all_data.append(row_data)

    # Compute permutation tests first if requested (needed to annotate plot)
    perm_results = {}  # (row, col) -> (obs_diff, p_value, reject)
    if args.permutation_test:
        print("\n--- Permutation Tests vs. Human Baseline (alpha=0.0167, Bonferroni-corrected) ---")
        for row_idx, (model_cfg, row_data) in enumerate(zip(MODELS, all_data)):
            for col_idx, ((temp_label, _), (df, _)) in enumerate(zip(model_cfg["variants"], row_data)):
                label = f"{model_cfg['name']} ({temp_label})"
                obs_diff, p_value, reject = run_permutation_test(df, human_df, label)
                perm_results[(row_idx, col_idx)] = (obs_diff, p_value, reject)

    # Global y-max across all series + human
    all_series = [series for row_data in all_data for (_, series) in row_data] + [human_series]
    global_max = max(
        max(int((s == v).sum()) for v in LIKERT_VALUES)
        for s in all_series
    )
    y_max = global_max + 20

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)

    for row_idx, (model_cfg, row_data) in enumerate(zip(MODELS, all_data)):
        color = model_cfg["color"]
        for col_idx, ((temp_label, _), (df, series)) in enumerate(zip(model_cfg["variants"], row_data)):
            ax = axes[row_idx, col_idx]
            title = f"{model_cfg['name']} ({temp_label})"
            p_val = perm_results.get((row_idx, col_idx))
            if p_val is not None:
                draw_bars(ax, series, color, title, p_value=p_val[1], reject=p_val[2])
            else:
                draw_bars(ax, series, color, title)
            draw_human_step(ax, human_series)
            ax.set_ylim(0, y_max)

    # Single legend from the first axis that has the human step handle
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels_leg,
        loc="lower center", ncol=1,
        fontsize=9, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    out = "figures/plots/distributions/post_stance_temperature_ablation"
    fig.savefig(f"{out}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"Saved {out}.svg / .png / .pdf")


main()
