import sys

import json
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
from belief_update_sim.ranking import ranks_from_model_ordering

# Set consistent font sizes (increased by 25%)
plt.rcParams.update({
    'font.size': 12.5,  # Base font size (was 10)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 12.5,  # Axis labels (was 10)
    'axes.titlesize': 13.75,  # Title (was 11)
    'xtick.labelsize': 11.25,  # X-tick labels (was 9)
    'ytick.labelsize': 11.25,  # Y-tick labels (was 9)
    'legend.fontsize': 11.25,  # Legend (was 9)
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 0.8,
    'axes.linewidth': 0.6,
})

PROLIFIC_DATA_DIR = 'data/prolific_data'

with open("colors.yaml") as _f:
    _yaml = yaml.safe_load(_f)
_COLORS = _yaml["models"]
HUMAN_COLOR = _yaml["human"]

# Reordered MODELS: Models first, then Human will be added separately
MODELS = {
    "GPT-5.2":                ("results/gpt5.2_results.xlsx",                                  "gpt5_2",           _COLORS["GPT-5.2"]),
    "GPT-5-mini":             ("results/academic-ai-gpt-5-mini_all_personas_results.xlsx",      "gpt5_mini",        _COLORS["GPT-5-mini"]),
    "Gemini-3-flash":         ("results/gemini-3-flash-preview_all_personas_results.xlsx",      "gemini_3_flash",   _COLORS["Gemini-3-flash"]),
    "Claude Opus 4.6":        ("results/claude-opus-4-6_all_personas_results.xlsx",             "claude_opus_4_6",  _COLORS["Claude Opus 4.6"]),
    "Qwen3-32B":              ("results/Qwen3_32B_all_personas_results.xlsx",                   "qwen3_32b",        _COLORS["Qwen3-32B"]),
    "Llama-3.3-70B-Instruct": ("results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",     "llama_70b",        _COLORS["Llama-3.3-70B-Instruct"]),
}


def load_ranking_data():
    # Load all LLM data files, indexed by (persona_id, topic)
    llm_indices = {}
    for _, (file_path, slug, _) in MODELS.items():
        data = load_normalized_data(file_path)
        data = data[['persona_id', 'topic', 'rank_1', 'rank_2', 'rank_3']]
        llm_indices[slug] = data.set_index(['persona_id', 'topic'])

    human_data = load_human_normalized_data()
    human_data = human_data[['persona_id', 'topic', 'package', 'message_1_shown',
                              'message_2_shown', 'message_3_shown',
                              'rank_message_1_shown', 'rank_message_2_shown', 'rank_message_3_shown']]
    human_index = human_data.set_index(['persona_id', 'topic'])

    rows = []
    for persona_id in os.listdir(PROLIFIC_DATA_DIR):
        if not os.path.isdir(os.path.join(PROLIFIC_DATA_DIR, persona_id)):
            continue
        study_path = os.path.join(PROLIFIC_DATA_DIR, persona_id, 'study_data.json')
        with open(study_path, encoding='utf-8') as f:
            study = json.load(f)

        for topic_entry in study['topics']:
            topic = topic_entry['topic']
            package = topic_entry['package']
            message_order_perm = topic_entry['message_order_perm']

            key = (persona_id, topic)

            # --- human data ---
            if key not in human_index.index:
                raise ValueError(f"No human data for persona_id={persona_id}, topic={topic}")
            h = human_index.loc[key]

            if h['package'] != package:
                raise ValueError(
                    f"Package mismatch for persona_id={persona_id}, topic={topic}: "
                    f"study_data={package!r}, human={h['package']!r}"
                )
            human_perm = [int(h['message_1_shown']), int(h['message_2_shown']), int(h['message_3_shown'])]
            if human_perm != message_order_perm:
                raise ValueError(
                    f"Message order mismatch for persona_id={persona_id}, topic={topic}: "
                    f"study_data={message_order_perm}, human={human_perm}"
                )

            row = {
                'persona_id': persona_id,
                'topic': topic,
                'statement_formulation': topic_entry['statement_formulation'],
                'package': package,
                'message_order_index': topic_entry['message_order_index'],
                'message_order_perm': message_order_perm,
                'shown_messages': topic_entry['shown_messages'],
                'message_1_human_rank': h['rank_message_1_shown'],
                'message_2_human_rank': h['rank_message_2_shown'],
                'message_3_human_rank': h['rank_message_3_shown'],
            }

            # --- llm data ---
            for slug, llm_index in llm_indices.items():
                if key not in llm_index.index:
                    raise ValueError(f"No LLM data ({slug}) for persona_id={persona_id}, topic={topic}")
                l = llm_index.loc[key]
                # The model reports an ordering (rank_i = id of the comment
                # placed i-th); the human columns hold per-slot ranks. Convert
                # so both sit in the same space before the remap below.
                ranks = ranks_from_model_ordering(l)
                row[f'message_1_{slug}_rank'] = ranks['rank_1']
                row[f'message_2_{slug}_rank'] = ranks['rank_2']
                row[f'message_3_{slug}_rank'] = ranks['rank_3']

            rows.append(row)

    df = pd.DataFrame(rows)

    # Convert local (display-position) ranks to global (source-message-id) ranks.
    # message_order_perm[i] = source_message_id shown at display position i+1, so:
    #   global rank of source message j = local rank at the position where j was shown.
    # Order: Models first, then Human
    slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
    for suffix in slugs:
        cols = [f'message_1_{suffix}_rank', f'message_2_{suffix}_rank', f'message_3_{suffix}_rank']
        global_ranks = {1: [], 2: [], 3: []}
        for _, row in df.iterrows():
            perm = row['message_order_perm']
            if suffix == 'human':
                shown_src_ids = [m['source_message_id'] for m in row['shown_messages']]
                if shown_src_ids != perm:
                    raise ValueError(
                        f"shown_messages source_message_ids {shown_src_ids} do not match "
                        f"message_order_perm {perm} for persona_id={row['persona_id']}, topic={row['topic']}"
                    )
            local = [row[c] for c in cols]
            src_to_rank = {src_id: local[i] for i, src_id in enumerate(perm)}
            for src_id in (1, 2, 3):
                global_ranks[src_id].append(src_to_rank[src_id])
        df[f'message_1_{suffix}_rank'] = global_ranks[1]
        df[f'message_2_{suffix}_rank'] = global_ranks[2]
        df[f'message_3_{suffix}_rank'] = global_ranks[3]

    return df


def get_average_message_ranks(df, print_output=False, return_agg=False):
    """Returns a dict {'group_a': mean_df, 'group_b': mean_df}.
    If return_agg=True, returns {'group_a': (mean_df, agg_df), 'group_b': (mean_df, agg_df)}."""
    # Order: Models first, then Human
    slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
    rank_cols = [f'message_{i}_{s}_rank' for s in slugs for i in (1, 2, 3)]

    personas = pd.Series(df['persona_id'].unique()).sample(frac=1, random_state=42).values
    mid = len(personas) // 2
    results = {}
    for key, label, ids in [('group_a', 'Group A', personas[:mid]),
                             ('group_b', 'Group B', personas[mid:])]:
        half = df[df['persona_id'].isin(ids)]
        mean_df = half.groupby(['topic', 'package'])[rank_cols].mean()

        agg_parts = []
        for s in slugs:
            cols = [f'message_1_{s}_rank', f'message_2_{s}_rank', f'message_3_{s}_rank']
            ranked = mean_df[cols].rank(axis=1, method='average').rename(columns={
                f'message_1_{s}_rank': f'message_1_{s}_agg',
                f'message_2_{s}_rank': f'message_2_{s}_agg',
                f'message_3_{s}_rank': f'message_3_{s}_agg',
            })
            agg_parts.append(ranked)
        agg_df = pd.concat(agg_parts, axis=1)

        if print_output:
            print(f'\n--- {label} (n={len(ids)} personas) ---')
            print(mean_df.to_string())
            if return_agg:
                print('\nAggregated ranking:')
                print(agg_df.to_string())

        results[key] = (mean_df, agg_df) if return_agg else mean_df

    return results


def analyse_rankings(data):
    # Order: Models first, then Human
    slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
    msg_cols = lambda s: [f'message_1_{s}_rank', f'message_2_{s}_rank', f'message_3_{s}_rank']

    mean_a, mean_b = data['group_a'], data['group_b']

    # Target: human Group B, flattened across (topic, package) x 3 messages
    target = mean_b[msg_cols('human')].values.flatten()

    predictors = {}
    # First add LLM models from Group B (models first)
    for slug in slugs[:-1]:  # All LLM slugs (excludes human)
        predictors[f'{slug}_b'] = mean_b[msg_cols(slug)].values.flatten()
    # Then add human_a
    predictors['human_a'] = mean_a[msg_cols('human')].values.flatten()

    rows_rank = []
    rows_cont = []
    for name, pred in predictors.items():
        mae = abs(pred - target).mean()

        rho, p_spearman = stats.spearmanr(pred, target)
        rows_rank.append({'predictor': name, 'spearman_rho': rho, 'p_spearman': p_spearman, 'mae': mae})

        r, p_pearson = stats.pearsonr(pred, target)
        rows_cont.append({'predictor': name, 'pearson_r': r, 'p_pearson': p_pearson, 'mae': mae})

    rank_results = pd.DataFrame(rows_rank).set_index('predictor').sort_values('spearman_rho', ascending=False)
    print('\n--- Rank-based measures (Spearman rho + MAE) ---')
    print(rank_results.to_string(float_format='{:.4f}'.format))

    human_a_errors = abs(predictors['human_a'] - target)
    print('\n--- Wilcoxon signed-rank test: human_a vs each LLM_b (H1: human_a errors smaller) ---')
    for slug in slugs[:-1]:  # LLM slugs only
        llm_errors = abs(predictors[f'{slug}_b'] - target)
        stat, p = stats.wilcoxon(human_a_errors, llm_errors, alternative='less')
        print(f"  human_a vs {slug}_b:  stat={stat:.1f},  p={p:.4f}")

    cont_results = pd.DataFrame(rows_cont).set_index('predictor').sort_values('pearson_r', ascending=False)
    print('\n--- Continuous measures (Pearson r + MAE) ---')
    print(cont_results.to_string(float_format='{:.4f}'.format))

    print('\n--- Paired t-test: human_a vs each LLM_b (H1: human_a errors smaller) ---')
    for slug in slugs[:-1]:  # LLM slugs only
        llm_errors = abs(predictors[f'{slug}_b'] - target)
        stat, p = stats.ttest_rel(human_a_errors, llm_errors, alternative='less')
        print(f"  human_a vs {slug}_b:  t={stat:.3f},  p={p:.4f}")

def get_extreme_comments_by_source(df):
    # Order: Models first, then Human
    slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
    source_labels = {slug: name for name, (_, slug, _) in MODELS.items()}
    source_labels['human'] = 'Human'

    # Overall mean ranks per (topic, package) across all personas
    rank_cols = [f'message_{i}_{s}_rank' for s in slugs for i in (1, 2, 3)]
    mean_df = df.groupby(['topic', 'package'])[rank_cols].mean()

    # Build text lookup: (topic, package, source_message_id) -> text
    text_lookup = {}
    for _, row in df.iterrows():
        for msg in row['shown_messages']:
            key = (row['topic'], row['package'], msg['source_message_id'])
            if key not in text_lookup:
                text_lookup[key] = msg['text']

    lines = []
    for s in slugs:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"SOURCE: {source_labels[s]}")
        lines.append(f"{'=' * 80}")

        records = []
        for (topic, package), row in mean_df.iterrows():
            for msg_id in (1, 2, 3):
                records.append({
                    'topic': topic,
                    'package': package,
                    'source_message_id': msg_id,
                    'mean_rank': row[f'message_{msg_id}_{s}_rank'],
                })
        records_df = pd.DataFrame(records)

        for label, idx_fn in [('MOST CONVINCING (lowest avg rank)',  'idxmin'),
                               ('LEAST CONVINCING (highest avg rank)', 'idxmax')]:
            rec = records_df.loc[getattr(records_df['mean_rank'], idx_fn)()]
            key = (rec['topic'], rec['package'], int(rec['source_message_id']))
            text = text_lookup.get(key, '[text not found]')
            lines.append(f"\n{label}")
            lines.append(f"Topic: {rec['topic']}  |  Package: {rec['package']}  |  "
                         f"Message ID: {int(rec['source_message_id'])}  |  Avg rank: {rec['mean_rank']:.4f}")
            lines.append(f"\n{text}\n")

    output = '\n'.join(lines)
    with open('figures/rankings/extreme_comments.txt', 'w', encoding='utf-8') as f:
        f.write(output)
    print("Written to rankings/extreme_comments.txt")


def plot_avg_comment_rankings(df):
    # Order: Models first, then Human
    slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
    colors = [color for _, _, color in MODELS.values()] + [HUMAN_COLOR]

    rank_cols = [f'message_{i}_{s}_rank' for s in slugs for i in (1, 2, 3)]
    mean_df = df.groupby(['topic', 'package'])[rank_cols].mean()

    distributions = [
        mean_df[[f'message_1_{s}_rank', f'message_2_{s}_rank', f'message_3_{s}_rank']].values.flatten()
        for s in slugs
    ]

    # Sized so the rankings panel's outer aspect (h/w â‰ˆ 0.44) is half the human
    # panel's outer aspect (h/w â‰ˆ 0.86 from figsize 3.2Ã—2.8). When the combined
    # figure places this panel at 2Ã— the human panel's width, both end up with
    # equal display heights â€” so their x-axis spines (bottom) line up, and the
    # top of their data areas lines up too.
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bp = ax.boxplot(distributions, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=1.5),
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                                                   markeredgecolor='black', markersize=5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # X-axis: hide tick labels (model identity comes from the master legend in the
    # combined figure). Keep the bottom spine and tick marks intact.
    ax.set_xticks(range(1, len(slugs) + 1))
    ax.set_xticklabels([])
    ax.set_ylabel('Mean rank', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=11.25)
    # A mean rank is bounded by [1, 3], and the models do reach both ends
    # (several comments are ranked first, or last, by every persona). Pin the
    # axis to the full attainable range rather than to the current data, so the
    # panel stays comparable across regenerations. A tighter window silently
    # clipped the boxes when the model ranks were corrected.
    ax.set_ylim(0.9, 3.1)
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
    ax.axhline(2, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    data_min = min(d.min() for d in distributions)
    data_max = max(d.max() for d in distributions)
    if data_min < 0.9 or data_max > 3.1:
        raise ValueError(f"mean ranks outside [1, 3]: {data_min:.3f}..{data_max:.3f}")
    
    # Legend removed - no legend added
    
    fig.tight_layout()
    fig.savefig('figures/rankings/plot_avg_comment_rankings.png', dpi=300)
    fig.savefig('figures/rankings/plot_avg_comment_rankings.pdf')
    fig.savefig('figures/rankings/plot_avg_comment_rankings.svg', format='svg')
    plt.close(fig)
    print("Written to rankings/plot_avg_comment_rankings.png / .pdf / .svg")


def print_sorted_mean_ranks(df):
    slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
    labels = list(MODELS.keys()) + ['Human']

    rank_cols = [f'message_{i}_{s}_rank' for s in slugs for i in (1, 2, 3)]
    mean_df = df.groupby(['topic', 'package'])[rank_cols].mean()

    print('\n--- Sorted mean ranks with boxplot boundaries ---')
    for label, slug in zip(labels, slugs):
        vals = np.sort(
            mean_df[[f'message_1_{slug}_rank', f'message_2_{slug}_rank', f'message_3_{slug}_rank']]
            .values.flatten()
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        wlo = vals[vals >= q1 - 1.5 * iqr].min()
        whi = vals[vals <= q3 + 1.5 * iqr].max()

        separators = sorted([
            (wlo, 'whisker low'),
            (q1,  'Q1'),
            (med, 'median'),
            (q3,  'Q3'),
            (whi, 'whisker high'),
        ])
        sep_idx = 0

        print(f'\n{label}')
        for i, v in enumerate(vals):
            while sep_idx < len(separators) and separators[sep_idx][0] <= v + 1e-9:
                sv, sname = separators[sep_idx]
                print(f'  â”€â”€ {sname} ({sv:.4f}) â”€â”€')
                sep_idx += 1
            print(f'  {i+1:2d}.  {v:.4f}')
        while sep_idx < len(separators):
            sv, sname = separators[sep_idx]
            print(f'  â”€â”€ {sname} ({sv:.4f}) â”€â”€')
            sep_idx += 1


df = load_ranking_data()
data = get_average_message_ranks(df, print_output=True)
analyse_rankings(data)
get_extreme_comments_by_source(df)
plot_avg_comment_rankings(df)
print_sorted_mean_ranks(df)

# Sanity check: grand mean rank should be ~2 for all sources if design is balanced
print("\n--- Grand mean rank and observation count per source ---")
print(f"  {'Source':30s}  {'mean':>8}  {'n obs':>8}")
# Order: Models first, then Human
slugs = [slug for _, slug, _ in MODELS.values()] + ['human']
labels = list(MODELS.keys()) + ['Human']
for label, slug in zip(labels, slugs):
    cols = [f'message_{i}_{slug}_rank' for i in (1, 2, 3)]
    series = df[cols].stack()
    print(f"  {label:30s}  mean={series.mean():.4f}  n={len(series):6d}")