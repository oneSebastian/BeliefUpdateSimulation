"""
Standalone combined figure: z-score plot (log scale) alongside 12 initial-stance
and post-stance distribution panels for 6 models (own-initial-belief ablation).

Layout: zscore on the left (spanning full height), 6 rows Ã— 2 columns of
distribution panels (init | post) on the right.

Run from: HumanSimulationProject/
Input:    all_models_combined_results.csv   (generate via data_analysis_bafa_meme.py)
          results/ablations/*.xlsx
Output:   figures/plots/zscore_initial_beliefs_combined.svg
"""

import sys

import os
import re
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS
from belief_update_sim.config import DERIVED_DIR

# â”€â”€ style â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica'],
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'lines.linewidth':    0.8,
    'axes.linewidth':     0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

OUT_DIR = "figures/plots"
os.makedirs(OUT_DIR, exist_ok=True)

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
MODEL_COLORS = _yaml["models"]
HUMAN_COLOR  = _yaml["human"]

# Zscore model colors (Human uses neutral grey; key names match MODEL_NAME_MAPPING values)
ZSCORE_COLORS = {
    'GPT-5.2':                '#8dd3c7',
    'Gemini-3-flash-preview': '#b15928',
    'Llama-3.3-70B-Instruct': '#bebada',
    'Claude-opus-4-6':        '#fb8072',
    'GPT-5-mini':             '#80b1d3',
    'Qwen3-32B':              '#fdb462',
    'Human':                  '#757575',
}

MODEL_NAME_MAPPING = {
    'GPT-5.2 Belief Shift':            'GPT-5.2',
    'Gemini Belief Shift':             'Gemini-3-flash-preview',
    'Human - Llama Bias (Difference)': 'Llama-3.3-70B-Instruct',
    'Claude Belief Shift':             'Claude-opus-4-6',
    'GPT-5 Mini Belief Shift':         'GPT-5-mini',
    'Human - Qwen Bias (Difference)':  'Qwen3-32B',
    'Human Belief Change':             'Human',
}

GROUPED_MODAL_MAP = {
    'topic':                  'UBI',
    'stance_direction':       'Pro',
    'gender':                 'Female',
    'ethnicity_grouped':      'White',
    'country_grouped':        'Great Britain',
    'student_grouped':        'Not currently a student',
    'employment_status':      'Employed full-time',
    'highest_qualification':  "Bachelor's degree",
}

ABLATION_MODELS = [
    ("Qwen3-32B",              "results/ablations/Qwen3-32B_own-initial-belief.xlsx"),
    ("Llama-3.3-70B-Instruct", "results/ablations/Llama-3.3-70B-Instruct_own-initial-belief.xlsx"),
    ("GPT-5-mini",             "results/ablations/academic-ai-gpt-5-mini_own-initial-belief.xlsx"),
    ("GPT-5.2",                "results/ablations/gpt-5.2_own-initial-belief.xlsx"),
    ("Gemini-3-flash",         "results/ablations/gemini-3-flash-preview_own-initial-belief.xlsx"),
    ("Claude Opus 4.6",        "results/ablations/claude-opus-4-6_own-initial-belief.xlsx"),
]

DISPLAY_ROWS = [
    ["GPT-5.2",    "GPT-5-mini"],
    ["Gemini-3-flash", "Claude Opus 4.6"],
    ["Qwen3-32B",  "Llama-3.3-70B-Instruct"],
]


LIKERT_VALUES = [-2, -1, 0, 1, 2]
LIKERT_LABELS = ["SD", "D", "N", "A", "SA"]

# â”€â”€ zscore helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _clean_label(term, gmap):
    if term == 'Intercept':
        return 'Intercept'
    if 'initial_belief' in term.lower():
        return 'Initial Belief'
    if 'familiarity' in term.lower():
        return 'Familiarity'
    if 'topic' in term.lower() and 'Penalty' in term:
        return 'Topic: Penalty (vs UBI)'
    if 'topic' in term.lower() and 'Weight Loss' in term:
        return 'Topic: Weight Loss (vs UBI)'

    pat = re.compile(r'^(?:C\()?([^\)\[]+)(?:\))?(?:\[(?:T\.)?(.+?)\])?$')
    m = pat.match(term)
    if not m:
        return term.replace('_centered', '').replace('C(', '').replace(')', '')

    var, level = m.group(1), m.group(2)
    if level is None:
        return var.replace('_centered', '')

    level = level.strip()
    if re.fullmatch(r'\.?\d+', level):
        if level.endswith('.0'):
            level = level[:-2]
        ref = gmap.get(var, 'reference') if gmap else 'reference'
        return f"{var} = {'Reference' if level == '0' else f'Other (vs {ref})'}"

    ref = gmap.get(var, 'reference') if gmap else 'reference'
    return f"{var} = {level} (vs {ref})"


def _wrap(text, width=38):
    words = text.split()
    lines, cur, n = [], [], 0
    for w in words:
        if n + len(w) + bool(cur) > width:
            lines.append(' '.join(cur))
            cur, n = [w], len(w)
        else:
            cur.append(w)
            n += len(w) + bool(cur) - 1
    if cur:
        lines.append(' '.join(cur))
    return '\n'.join(lines)


def _set_symlog(ax, lo=-100, hi=100):
    def fwd(x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.log10(np.abs(x) + 1)

    def inv(y):
        y = np.asarray(y, dtype=float)
        return np.sign(y) * (10 ** np.abs(y) - 1)

    ax.set_xscale('function', functions=(fwd, inv))
    major = [t for t in [-100, -10, -1, 0, 1, 10, 100] if lo <= t <= hi]
    ax.set_xticks(major)
    ax.set_xticklabels([str(t) for t in major])

    minor = (
        list(range(-90, -9, 10)) +
        [n for n in range(-9, 0) if n not in major] +
        [n for n in range(2, 10) if n not in major] +
        list(range(20, 100, 10))
    )
    minor = [t for t in minor if lo <= t <= hi]
    if minor:
        ax.set_xticks(minor, minor=True)
    else:
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(lo, hi)


def prepare_zscore_data(df, threshold=2.0):
    df = df[df['Variable'] != 'Intercept'].copy()
    mask = (
        df.groupby('Variable')['Z_value']
          .transform(lambda s: (s.abs() > threshold).any())
          .astype(bool)
    )
    df = df[mask].copy()
    models = df['Model'].unique().tolist()
    pivot = df.pivot_table(index='Variable', columns='Model', values='Z_value', aggfunc='first')
    pivot.index = [_wrap(_clean_label(v, GROUPED_MODAL_MAP)) for v in pivot.index]
    pivot = pivot.sort_index()
    return pivot, models


def draw_zscore(ax, z_df, models, threshold=2.0):
    predictors = z_df.index[::-1]
    y_pos = np.arange(len(predictors)) * 0.33

    for model in models:
        if model not in z_df.columns:
            continue
        name   = MODEL_NAME_MAPPING.get(model, model)
        color  = ZSCORE_COLORS.get(name, '#333333')
        marker = 'D' if name == 'Human' else 'o'
        size   = 130 if name == 'Human' else 110
        ax.scatter(z_df[model].values[::-1], y_pos,
                   label=name, color=color, marker=marker,
                   s=size, alpha=0.9, edgecolors='white', linewidth=1.2, zorder=3)

    ax.axvline(0, color='black', linewidth=1.5, zorder=1)
    ax.axvline( threshold, color='red', linestyle='--', linewidth=1.5,
                label=f'|z|={threshold}', zorder=1)
    ax.axvline(-threshold, color='red', linestyle='--', linewidth=1.5, zorder=1)
    ax.axvspan(-threshold, threshold, alpha=0.08, color='gray', zorder=0)

    _set_symlog(ax)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors, fontsize=9)
    ax.set_xlabel('z-score', fontsize=12)
    ax.set_ylabel('Predictor', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(True, axis='x', which='major', alpha=0.35, linestyle='-',  linewidth=0.6)
    ax.grid(True, axis='x', which='minor', alpha=0.15, linestyle=':', linewidth=0.4)
    ax.set_axisbelow(True)

    legend = ax.legend(
        loc='lower left', bbox_to_anchor=(0.02, 0.02),
        frameon=True, fontsize=8, framealpha=0.9,
        title='Models', title_fontsize=9,
        ncol=1, handleheight=1.5, handletextpad=0.5,
        borderpad=0.8, labelspacing=0.8,
    )
    legend.get_frame().set_linewidth(0.5)
    legend.get_frame().set_boxstyle('square', pad=0.5)

# â”€â”€ distribution helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_human():
    df = load_human_normalized_data()
    init = pd.to_numeric(df["init_stance"], errors="coerce").dropna().astype(int)
    post = pd.to_numeric(df["new_belief"],  errors="coerce").dropna().astype(int)
    return init, post


def load_ablation(path):
    df = load_normalized_data(path)
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "init_stance"] *= -1
    init = pd.to_numeric(df["init_stance"], errors="coerce").dropna().astype(int)
    post = pd.to_numeric(df["new_belief"],  errors="coerce").dropna().astype(int)
    return init, post


def draw_dist(ax, series, color, title, human_series, human_label, y_max):
    total  = len(series)
    counts = [int((series == v).sum()) for v in LIKERT_VALUES]
    x      = np.arange(len(LIKERT_VALUES))

    ax.bar(x, counts, color=color, edgecolor='white', linewidth=0.4,
           width=0.7, zorder=2, alpha=0.85)

    h_counts = [int((human_series == v).sum()) for v in LIKERT_VALUES]
    edges    = np.arange(-0.5, len(LIKERT_VALUES) + 0.5)
    ax.stairs(h_counts, edges, color=HUMAN_COLOR, lw=1.4, zorder=5, label=human_label)

    ax.set_title(title, fontsize=12, fontweight='normal', pad=3, color=color, loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(LIKERT_LABELS, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(3))
    ax.set_xlim(-0.6, len(LIKERT_VALUES) - 0.4)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4, zorder=0)

# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    # â”€â”€ zscore data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    csv_path = Path(f"{DERIVED_DIR}/all_models_combined_results.csv")
    if not csv_path.exists():
        raise SystemExit(
            "all_models_combined_results.csv not found.\n"
            "Run the results-extraction section of data_analysis_bafa_meme.py first."
        )
    df_z = pd.read_csv(csv_path)
    z_df, models = prepare_zscore_data(df_z, threshold=2.0)
    print(f"Z-score predictors selected: {len(z_df)}")

    # â”€â”€ distribution data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    human_init, human_post = load_human()
    dist_cache, all_series = [], [human_init, human_post]
    for label, path in ABLATION_MODELS:
        init, post = load_ablation(path)
        dist_cache.append((label, init, post))
        all_series.extend([init, post])
        print(f"{label}: init n={len(init)}, post n={len(post)}")

    global_max = max(
        max(int((s == v).sum()) for v in LIKERT_VALUES)
        for s in all_series
    )
    y_max = global_max + 5
    print(f"Shared y-max: {y_max}")

    # â”€â”€ figure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Distribution grid: 3 rows Ã— 4 columns (pairs: init|post per model, 2 models per row)
    # Row 0: GPT-5.2 init, GPT-5.2 post, GPT-5-mini init, GPT-5-mini post
    # Row 1: Gemini-3-flash init|post, Claude Opus 4.6 init|post
    # Row 2: Qwen3-32B init|post, Llama init|post
    fig = plt.figure(figsize=(20, 10))
    gs  = GridSpec(
        3, 5, figure=fig,
        width_ratios=[3, 1.1, 1.1, 1.1, 1.1],
        hspace=0.20, wspace=0.10,
        left=0.04, right=0.97, top=0.92, bottom=0.05,
    )

    # Zscore panel (left column, full height)
    ax_z = fig.add_subplot(gs[:, 0])
    draw_zscore(ax_z, z_df, models)
    ax_z.text(-0.08, 1.01, 'a', transform=ax_z.transAxes,
              fontsize=12, fontweight='bold', va='bottom', ha='left')

    # Distribution panels in 3Ã—4 grid (columns 1â€“4)
    dist_lookup = {label: (init, post) for label, init, post in dist_cache}
    panel_letter = 1  # 'a'=0, so b=1

    for row, model_pair in enumerate(DISPLAY_ROWS):
        for col_pair, label in enumerate(model_pair):
            init, post = dist_lookup[label]
            color    = MODEL_COLORS[label]
            init_col = 1 + col_pair * 2   # cols 1 and 3
            post_col = 2 + col_pair * 2   # cols 2 and 4

            ax_i = fig.add_subplot(gs[row, init_col])
            draw_dist(ax_i, init, color,
                      label, human_init, "Human Initial Stance", y_max)
            ax_i.text(-0.08, 1.01, chr(ord('a') + panel_letter),
                      transform=ax_i.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', ha='left')
            panel_letter += 1
            if col_pair > 0:
                ax_i.set_ylabel('')
                ax_i.tick_params(labelleft=False)

            ax_p = fig.add_subplot(gs[row, post_col])
            draw_dist(ax_p, post, color,
                      label, human_post, "Human Post-Stance", y_max)
            ax_p.text(-0.08, 1.01, chr(ord('a') + panel_letter),
                      transform=ax_p.transAxes,
                      fontsize=12, fontweight='bold', va='bottom', ha='left')
            panel_letter += 1
            ax_p.set_ylabel('')
            ax_p.tick_params(labelleft=False)

            # Column headers above first row only
            if row == 0:
                ax_i.annotate(
                    "Initial Stance", xy=(0.5, 1.08), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333',
                )
                ax_p.annotate(
                    "Post-Stance", xy=(0.5, 1.08), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333',
                )

    out = f"{OUT_DIR}/zscore_initial_beliefs_combined.svg"
    fig.savefig(out, format='svg', bbox_inches='tight')
    print(f"\nSaved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
