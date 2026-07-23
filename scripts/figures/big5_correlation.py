"""
Point 1: Do Big5 personality traits predict Î” belief in humans?
         Do LLMs with full persona replicate that correlation?
         Does removing persona break it?

Visualisation: heatmap of Spearman Ï (Big5 trait Ã— data source).
Rows grouped by model; humans shown as reference row.
Significant correlations (p < 0.05) are annotated with *.

Run from: HumanSimulationProject/
Output:   claude/big5_correlation.png / .pdf / .eps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr
import os
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# Set Nature Communications style parameters
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5
})

# Create output directory
os.makedirs("figures/plots", exist_ok=True)

# â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


BIG5 = {
    "BIG5_openness_score":           "Openness",
    "BIG5_conscientiousness_score":  "Conscientiousness",
    "BIG5_extraversion_score":       "Extraversion",
    "BIG5_agreeableness_score":      "Agreeableness",
    "BIG5_neuroticism_score":        "Neuroticism",
}

MODELS = {
    "Qwen3-32B": {
        "base":           "results/Qwen3_32B_all_personas_results.xlsx",
        "no-persona":     "results/ablations/Qwen3-32B_no-persona.xlsx",
        "no-personality": "results/ablations/Qwen3-32B_no-personality.xlsx",
        "no-demographic": "results/ablations/Qwen3-32B_no-demographic.xlsx",
    },
    "Llama-3.3-70B-Instruct": {
        "base":           "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "no-persona":     "results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx",
        "no-personality": "results/ablations/Llama-3.3-70B-Instruct_no-personality.xlsx",
        "no-demographic": "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx",
    },
    "GPT-5-mini": {
        "base":           "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
        "no-persona":     "results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx",
        "no-personality": "results/ablations/academic-ai-gpt-5-mini_no-personality.xlsx",
        "no-demographic": "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx",
    },
}

CONDITION_LABELS = {
    "base":           "Base (full persona)",
    "no-persona":     "No persona",
    "no-personality": "No personality traits",
    "no-demographic": "No demographic info",
}

# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_big5_from_source() -> pd.DataFrame:
    """
    Compute correct BFI-10 scores from raw oTree item responses.
    BFI-10 scoring (R = reversed, scale stored as âˆ’2â€¦+2):
      Extraversion:      1R, 6   â†’ (âˆ’big1 + big6)  / 2 + 3
      Agreeableness:     2,  7R  â†’ ( big2 âˆ’ big7)  / 2 + 3
      Conscientiousness: 3R, 8   â†’ (âˆ’big3 + big8)  / 2 + 3
      Neuroticism:       4R, 9   â†’ (âˆ’big4 + big9)  / 2 + 3
      Openness:          5R, 10  â†’ (âˆ’big5 + big10) / 2 + 3
    """
    src = pd.read_csv("data/cleaned_for_llm_391_participant_otree.csv")
    b = {i: src[f"survey.1.player.big{i}"] for i in range(1, 11)}
    src["BIG5_extraversion_score"]      = (-b[1]  + b[6])  / 2 + 3
    src["BIG5_agreeableness_score"]     = ( b[2]  - b[7])  / 2 + 3
    src["BIG5_conscientiousness_score"] = (-b[3]  + b[8])  / 2 + 3
    src["BIG5_neuroticism_score"]       = (-b[4]  + b[9])  / 2 + 3
    src["BIG5_openness_score"]          = (-b[5]  + b[10]) / 2 + 3
    return src.rename(columns={"participant.label": "persona_id"})


def load_human() -> pd.DataFrame:
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    # Attach correctly scored Big5 from source oTree data
    big5_src = load_big5_from_source()
    big5_cols = list(BIG5.keys()) + ["persona_id"]
    df = df.drop(columns=[c for c in BIG5.keys() if c in df.columns], errors='ignore')
    df = df.merge(big5_src[big5_cols].drop_duplicates("persona_id"), on="persona_id", how="left")
    return df


def load_llm_with_big5(path: str, human_df: pd.DataFrame) -> pd.DataFrame:
    """Load LLM results, compute delta, join Big5 scores from human data."""
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    # Big5 scores are constant per persona; take unique personaâ†’Big5 mapping
    big5_cols = list(BIG5.keys())
    persona_big5 = human_df[["persona_id"] + big5_cols].drop_duplicates("persona_id")
    df = df.merge(persona_big5, on="persona_id", how="inner")
    return df.dropna(subset=["delta"] + big5_cols)


# â”€â”€ correlation computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_rhos(df: pd.DataFrame, target: str = "delta") -> dict:
    """Spearman Ï and p-value for each Big5 trait vs target column."""
    results = {}
    for col, label in BIG5.items():
        sub = df[[col, target]].dropna()
        if len(sub) > 1:  # Need at least 2 points for correlation
            rho, pval = spearmanr(sub[col], sub[target])
        else:
            rho, pval = np.nan, 1.0
        results[label] = (rho, pval)
    return results


def save_figure(fig, out_stem):
    """Save figure in multiple formats."""
    for ext in ['.png', '.pdf', '.eps']:
        fig.savefig(f"{out_stem}{ext}", dpi=300, bbox_inches='tight')
    print(f"Saved {out_stem}.png / .pdf / .eps")


# â”€â”€ plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot():
    print("Loading human data...")
    human_df = load_human()

    # main rows: Human + LLM conditions
    rows = [("Human", "Human", compute_rhos(human_df))]
    for model_name, paths in MODELS.items():
        for cond, path in paths.items():
            print(f"Processing {model_name} - {cond}...")
            try:
                df = load_llm_with_big5(path, human_df)
                rows.append((CONDITION_LABELS[cond], model_name, compute_rhos(df)))
            except Exception as e:
                print(f"  Warning: Could not process {path}: {e}")
                # Add row with NaNs
                nan_results = {label: (np.nan, 1.0) for label in BIG5.values()}
                rows.append((CONDITION_LABELS[cond], model_name, nan_results))

    # extra row: Big5 â†’ human initial stance
    init_rhos = compute_rhos(human_df, target="initial_belief_normalized")
    init_row  = ("Init. stance", "Human init. stance", init_rhos)

    all_rows = rows + [init_row]

    trait_labels = list(BIG5.values())

    rho_matrix  = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    pval_matrix = np.array([[v[1] for v in r[2].values()] for r in all_rows])

    # â”€â”€ figure - optimized for Nature Communications â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_rows, n_cols = rho_matrix.shape
    # Width: 7" for double column, height scaled proportionally
    fig, ax = plt.subplots(figsize=(7, 8))

    vmax = 0.25
    im = ax.imshow(rho_matrix, cmap=plt.cm.RdBu, vmin=-vmax, vmax=vmax,
                   aspect="auto", zorder=0)

    # Add values with better visibility
    for i in range(n_rows):
        for j in range(n_cols):
            rho  = rho_matrix[i, j]
            pval = pval_matrix[i, j]
            
            # Skip NaN values
            if np.isnan(rho):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7,
                        color="gray", fontweight="normal")
                continue
            
            # Determine significance threshold
            thr = 0.05 if i == n_rows - 1 else (0.05 / (n_rows * n_cols))
            sig = "*" if pval < thr else ""
            txt = f"{rho:+.2f}{sig}"
            
            # Determine text color based on cell brightness
            brightness = abs(rho) / vmax if not np.isnan(rho) else 0.5
            color = "white" if brightness > 0.55 else "black"
            
            # Add background box for better readability on dark cells
            if brightness > 0.55:
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if sig else "normal",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="none", 
                                 edgecolor="none"))
            else:
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if sig else "normal")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(trait_labels, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in all_rows], fontsize=7)

    # group separators with thinner lines
    prev_group = None
    for i, (_, group, _) in enumerate(all_rows):
        if group != prev_group and i > 0:
            lw = 1.5 if group == "Human init. stance" else 1
            ax.axhline(i - 0.5, color="white", lw=lw, zorder=1)
        prev_group = group

    # Compact colorbar
    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.15)
    cb.set_label("Spearman Ï", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Minimal title for Nature Communications
    bonf_thr = 0.05 / (len(rows) * len(BIG5))
    ax.set_title(f"Big5 â†’ Belief Change: Spearman Ï\n* p < {bonf_thr:.4f} (Bonferroni)", 
                 fontsize=9, pad=8, fontweight="normal")

    plt.tight_layout()
    save_figure(fig, "figures/plots/big5_correlation")
    print(f"Bonferroni threshold: {bonf_thr:.4f}")


if __name__ == "__main__":
    plot()