"""
Big5 and demographic Spearman Ï / ÏÂ² heatmaps comparing 6 LLMs (base condition).
Rows: Human + 6 models + Init. belief (human).
Produces 4 plots:
  claude/big5_correlation_models.png
  claude/big5_effect_size_models.png
  claude/demographic_correlation_models.png
  claude/demographic_effect_size_models.png

Run from: HumanSimulationProject/
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


MODELS = {
    "GPT-5.2":                "results/gpt5.2_results.xlsx",
    "GPT-5-mini":             "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
    "Claude Opus 4.6":        "results/claude-opus-4-6_all_personas_results.xlsx",
    "Gemini-3-flash":         "results/gemini-3-flash-preview_all_personas_results.xlsx",
    "Qwen3-32B":              "results/Qwen3_32B_all_personas_results.xlsx",
    "Llama-3.3-70B-Instruct": "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
}

BIG5 = {
    "BIG5_openness_score":          "Openness",
    "BIG5_conscientiousness_score": "Conscientiousness",
    "BIG5_extraversion_score":      "Extraversion",
    "BIG5_agreeableness_score":     "Agreeableness",
    "BIG5_neuroticism_score":       "Neuroticism",
}

ORDINAL_VARS = {
    "age":                   "Age",
    "highest_qualification": "Education",
}
NOMINAL_VARS = {
    "gender_raw":        "Gender",
    "employment_status": "Employment",
}

# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def normalize(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def load_big5_from_source():
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


def load_human():
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    df = df.rename(columns={"gender": "gender_raw"})
    big5_src = load_big5_from_source()
    big5_cols = list(BIG5.keys())
    df = df.drop(columns=[c for c in big5_cols if c in df.columns])
    df = df.merge(big5_src[["persona_id"] + big5_cols].drop_duplicates("persona_id"),
                  on="persona_id", how="left")
    return df


def load_llm_base(path):
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    return df.dropna(subset=["delta"])


def attach_big5(llm_df, human_df):
    big5_cols = list(BIG5.keys())
    persona_big5 = human_df[["persona_id"] + big5_cols].drop_duplicates("persona_id")
    return llm_df.merge(persona_big5, on="persona_id", how="inner")


def attach_demog(llm_df):
    def extract(s):
        d = json.loads(s)
        return pd.Series({
            "age":                   d.get("age"),
            "gender_raw":            d.get("gender_raw"),
            "highest_qualification": d.get("highest_qualification_raw"),
            "employment_status":     d.get("employment_status_raw"),
        })
    demog = llm_df["demographic"].apply(extract)
    return pd.concat([llm_df[["delta"]], demog], axis=1)


# â”€â”€ correlation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_rhos(df, col_map, target="delta"):
    out = {}
    for col, label in col_map.items():
        sub = df[[col, target]].dropna()
        if len(sub) < 10:
            out[label] = (np.nan, np.nan)
        else:
            r, p = spearmanr(sub[col], sub[target])
            out[label] = (r, p)
    return out


def compute_kw(df, col_map, target="delta"):
    """Kruskal-Wallis ÎµÂ² for nominal variables."""
    out = {}
    for col, label in col_map.items():
        sub = df[[col, target]].dropna()
        if len(sub) < 10:
            out[label] = (np.nan, np.nan)
            continue
        groups = [g[target].values for _, g in sub.groupby(col) if len(g) >= 2]
        if len(groups) < 2:
            out[label] = (np.nan, np.nan)
            continue
        N = sum(len(g) for g in groups)
        H, p = kruskal(*groups)
        out[label] = (H / (N - 1), p)
    return out


# â”€â”€ heatmap renderers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_rho_heatmap(all_rows, col_labels, bonf_thr, n_main, title, outpath, figsize):
    rho_mat  = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    pval_mat = np.array([[v[1] for v in r[2].values()] for r in all_rows])

    n_rows, n_cols = rho_mat.shape
    vmax = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rho_mat, cmap=plt.cm.RdBu, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            rho  = rho_mat[i, j]
            pval = pval_mat[i, j]
            if np.isnan(rho):
                ax.text(j, i, "â€”", ha="center", va="center", fontsize=9)
                continue
            thr = 0.05 if i == n_rows - 1 else bonf_thr
            sig = "*" if pval < thr else ""
            txt = f"{rho:+.2f}{sig}"
            color = "white" if abs(rho) / vmax > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if sig else "normal")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in all_rows], fontsize=9)

    prev_group = None
    for i, (_, group, _) in enumerate(all_rows):
        if group != prev_group and i > 0:
            lw = 3 if i == n_main else 2
            ax.axhline(i - 0.5, color="white", lw=lw)
        prev_group = group

    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.04)
    cb.set_label("Spearman Ï", fontsize=9)
    ax.set_title(
        f"{title}\n"
        f"* Bonferroni-corrected p < {bonf_thr:.4f}  "
        f"(bottom row: p < 0.05 vs initial stance)",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close()


def draw_r2_heatmap(all_rows, col_labels, bonf_thr, n_main, title, outpath, figsize):
    rho_mat  = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    pval_mat = np.array([[v[1] for v in r[2].values()] for r in all_rows])
    r2_mat   = rho_mat ** 2 * 100

    n_rows, n_cols = rho_mat.shape
    vmax = 5.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.where(np.isnan(r2_mat), -1, r2_mat),
                   cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            rho  = rho_mat[i, j]
            r2   = r2_mat[i, j]
            pval = pval_mat[i, j]
            if np.isnan(rho):
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1,
                             color="lightgrey", zorder=2))
                ax.text(j, i, "â€”", ha="center", va="center",
                        fontsize=9, zorder=3)
                continue
            thr  = 0.05 if i == n_rows - 1 else bonf_thr
            sig  = "*" if pval < thr else ""
            sign = "+" if rho >= 0 else "âˆ’"
            txt  = f"{sign}{r2:.2f}%{sig}"
            color = "white" if r2 / vmax > 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if sig else "normal",
                    zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in all_rows], fontsize=9)

    prev_group = None
    for i, (_, group, _) in enumerate(all_rows):
        if group != prev_group and i > 0:
            lw = 3 if i == n_main else 2
            ax.axhline(i - 0.5, color="white", lw=lw)
        prev_group = group

    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.04)
    cb.set_label("ÏÂ²  (% variance explained)", fontsize=9)
    cb.ax.set_title(f"0â€“{vmax:.0f}%", fontsize=8, pad=4)
    ax.set_title(
        f"{title}\n"
        f"sign = direction of Ï   |   "
        f"* Bonferroni-corrected p < {bonf_thr:.4f}  "
        f"(bottom row: p < 0.05 vs initial stance)",
        fontsize=9, pad=10,
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close()


# â”€â”€ two-panel demographic renderer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_demog_split(all_rows, n_main, bonf_thr, title_suffix, rho_outpath, r2_outpath):
    """Side-by-side heatmap: Spearman Ï (or ÏÂ²) for ordinal + KW ÎµÂ² for nominal."""
    rho_mat = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    rho_p   = np.array([[v[1] for v in r[2].values()] for r in all_rows])
    kw_mat  = np.array([[v[0] for v in r[3].values()] for r in all_rows])
    kw_p    = np.array([[v[1] for v in r[3].values()] for r in all_rows])
    r2_mat  = rho_mat ** 2 * 100

    n_rows  = len(all_rows)
    vmax_rho = 0.25
    vmax_r2  = 5.0
    vmax_kw  = 0.05

    def _separators(ax):
        prev_group = None
        for i, (_, group, *_) in enumerate(all_rows):
            if group != prev_group and i > 0:
                lw = 3 if i == n_main else 2
                ax.axhline(i - 0.5, color="white", lw=lw)
            prev_group = group

    def _group_labels(ax, n_cols):
        drawn = set()
        for i, (_, group, *_) in enumerate(all_rows):
            if group not in drawn:
                indices = [j for j, (_, g, *_) in enumerate(all_rows) if g == group]
                mid = (indices[0] + indices[-1]) / 2
                ax.annotate(group,
                            xy=(n_cols - 0.5, mid), xycoords="data",
                            xytext=(n_cols + 0.2, mid), textcoords="data",
                            fontsize=8.5, va="center", annotation_clip=False)
                drawn.add(group)

    def _kw_cells(ax, mat, pmat, vmax):
        for i in range(n_rows):
            for j in range(mat.shape[1]):
                v, p = mat[i, j], pmat[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, color="lightgrey", zorder=2))
                    ax.text(j, i, "â€”", ha="center", va="center", fontsize=9, zorder=3)
                    continue
                thr = 0.05 if i == n_rows - 1 else bonf_thr
                sig = "*" if p < thr else ""
                color = "white" if v / vmax > 0.65 else "black"
                ax.text(j, i, f"{v:.3f}{sig}", ha="center", va="center", fontsize=9,
                        color=color, fontweight="bold" if sig else "normal", zorder=3)

    # â”€â”€ Spearman Ï plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(8, 5),
        gridspec_kw={"width_ratios": [2, 2], "wspace": 0.5},
    )
    im_l = ax_l.imshow(rho_mat, cmap=plt.cm.RdBu, vmin=-vmax_rho, vmax=vmax_rho, aspect="auto")
    for i in range(n_rows):
        for j in range(len(ORDINAL_VARS)):
            rho, p = rho_mat[i, j], rho_p[i, j]
            if np.isnan(rho):
                ax_l.text(j, i, "â€”", ha="center", va="center", fontsize=9); continue
            thr = 0.05 if i == n_rows - 1 else bonf_thr
            sig = "*" if p < thr else ""
            color = "white" if abs(rho) / vmax_rho > 0.55 else "black"
            ax_l.text(j, i, f"{rho:+.2f}{sig}", ha="center", va="center", fontsize=9,
                      color=color, fontweight="bold" if sig else "normal")
    ax_l.set_xticks(range(len(ORDINAL_VARS)))
    ax_l.set_xticklabels(list(ORDINAL_VARS.values()), fontsize=10)
    ax_l.set_yticks(range(n_rows))
    ax_l.set_yticklabels([r[0] for r in all_rows], fontsize=9)
    _separators(ax_l)
    cb_l = plt.colorbar(im_l, ax=ax_l, shrink=0.55, pad=0.04)
    cb_l.set_label("Spearman Ï", fontsize=9)
    ax_l.set_title("Ordinal (Spearman Ï)", fontsize=9)

    im_r = ax_r.imshow(np.where(np.isnan(kw_mat), -1, kw_mat),
                       cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax_kw, aspect="auto")
    _kw_cells(ax_r, kw_mat, kw_p, vmax_kw)
    ax_r.set_xticks(range(len(NOMINAL_VARS)))
    ax_r.set_xticklabels(list(NOMINAL_VARS.values()), fontsize=10)
    ax_r.set_yticks(range(n_rows))
    ax_r.set_yticklabels([], fontsize=9)
    _separators(ax_r)
    _group_labels(ax_r, len(NOMINAL_VARS))
    cb_r = plt.colorbar(im_r, ax=ax_r, shrink=0.55, pad=0.04)
    cb_r.set_label("KW ÎµÂ²", fontsize=9)
    cb_r.ax.set_title(f"0â€“{vmax_kw:.0%}", fontsize=8, pad=4)
    ax_r.set_title("Nominal (Kruskal-Wallis ÎµÂ²)", fontsize=9)
    fig.suptitle(
        f"Demographic Attributes â†’ Belief Change (Î”):  {title_suffix}\n"
        f"* Bonferroni-corrected p < {bonf_thr:.4f}  (bottom row: p < 0.05 vs initial stance)",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(rho_outpath, dpi=150, bbox_inches="tight")
    plt.savefig(rho_outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {rho_outpath}")
    plt.close()

    # â”€â”€ ÏÂ² / ÎµÂ² effect-size plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(8, 5),
        gridspec_kw={"width_ratios": [2, 2], "wspace": 0.5},
    )
    im_l = ax_l.imshow(np.where(np.isnan(r2_mat), -1, r2_mat),
                       cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax_r2, aspect="auto")
    for i in range(n_rows):
        for j in range(len(ORDINAL_VARS)):
            rho, r2, p = rho_mat[i, j], r2_mat[i, j], rho_p[i, j]
            if np.isnan(rho):
                ax_l.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, color="lightgrey", zorder=2))
                ax_l.text(j, i, "â€”", ha="center", va="center", fontsize=9, zorder=3); continue
            thr  = 0.05 if i == n_rows - 1 else bonf_thr
            sig  = "*" if p < thr else ""
            sign = "+" if rho >= 0 else "âˆ’"
            color = "white" if r2 / vmax_r2 > 0.65 else "black"
            ax_l.text(j, i, f"{sign}{r2:.2f}%{sig}", ha="center", va="center", fontsize=9,
                      color=color, fontweight="bold" if sig else "normal", zorder=3)
    ax_l.set_xticks(range(len(ORDINAL_VARS)))
    ax_l.set_xticklabels(list(ORDINAL_VARS.values()), fontsize=10)
    ax_l.set_yticks(range(n_rows))
    ax_l.set_yticklabels([r[0] for r in all_rows], fontsize=9)
    _separators(ax_l)
    cb_l = plt.colorbar(im_l, ax=ax_l, shrink=0.55, pad=0.04)
    cb_l.set_label("ÏÂ²  (% variance explained)", fontsize=9)
    cb_l.ax.set_title(f"0â€“{vmax_r2:.0f}%", fontsize=8, pad=4)
    ax_l.set_title("Ordinal (ÏÂ²)", fontsize=9)

    im_r = ax_r.imshow(np.where(np.isnan(kw_mat), -1, kw_mat),
                       cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax_kw, aspect="auto")
    _kw_cells(ax_r, kw_mat, kw_p, vmax_kw)
    ax_r.set_xticks(range(len(NOMINAL_VARS)))
    ax_r.set_xticklabels(list(NOMINAL_VARS.values()), fontsize=10)
    ax_r.set_yticks(range(n_rows))
    ax_r.set_yticklabels([], fontsize=9)
    _separators(ax_r)
    _group_labels(ax_r, len(NOMINAL_VARS))
    cb_r = plt.colorbar(im_r, ax=ax_r, shrink=0.55, pad=0.04)
    cb_r.set_label("KW ÎµÂ²  (variance explained)", fontsize=9)
    cb_r.ax.set_title(f"0â€“{vmax_kw:.0%}", fontsize=8, pad=4)
    ax_r.set_title("Nominal (Kruskal-Wallis ÎµÂ²)", fontsize=9)
    fig.suptitle(
        f"Demographic Attributes â†’ Belief Change (Î”):  Effect Sizes\n"
        f"sign = direction of Ï (ordinal only)   |   "
        f"* Bonferroni-corrected p < {bonf_thr:.4f}  (bottom row: p < 0.05 vs initial stance)",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(r2_outpath, dpi=150, bbox_inches="tight")
    plt.savefig(r2_outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {r2_outpath}")
    plt.close()


# â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run():
    human_df = load_human()

    # â”€â”€ Big5 rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    big5_rows = [("Human", "Human", compute_rhos(human_df, BIG5))]
    for model_name, path in MODELS.items():
        df = attach_big5(load_llm_base(path), human_df)
        big5_rows.append((model_name, model_name, compute_rhos(df, BIG5)))

    n_big5_main = len(big5_rows)
    bonf_big5   = 0.05 / (n_big5_main * len(BIG5))
    init_big5   = ("Init. belief", "Human init. belief",
                   compute_rhos(human_df, BIG5, target="initial_belief_normalized"))
    all_big5    = big5_rows + [init_big5]

    # â”€â”€ Demographic rows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    n_cols_demog = len(ORDINAL_VARS) + len(NOMINAL_VARS)
    demog_rows = [("Human", "Human",
                   compute_rhos(human_df, ORDINAL_VARS),
                   compute_kw(human_df, NOMINAL_VARS))]
    for model_name, path in MODELS.items():
        df = attach_demog(load_llm_base(path))
        demog_rows.append((model_name, model_name,
                           compute_rhos(df, ORDINAL_VARS),
                           compute_kw(df, NOMINAL_VARS)))

    n_demog_main = len(demog_rows)
    bonf_demog   = 0.05 / (n_demog_main * n_cols_demog)
    init_demog   = ("Init. belief", "Human init. belief",
                    compute_rhos(human_df, ORDINAL_VARS, target="initial_belief_normalized"),
                    compute_kw(human_df, NOMINAL_VARS,   target="initial_belief_normalized"))
    all_demog    = demog_rows + [init_demog]

    # â”€â”€ draw â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    draw_rho_heatmap(
        all_big5, list(BIG5.values()), bonf_big5, n_big5_main,
        "Big5 Personality â†’ Belief Change (Î”):  Spearman Ï  â€”  6 Models",
        "figures/plots/big5_correlation_models.png", figsize=(8, 5),
    )
    draw_r2_heatmap(
        all_big5, list(BIG5.values()), bonf_big5, n_big5_main,
        "Big5 Personality â†’ Belief Change (Î”):  Variance Explained (ÏÂ²)  â€”  6 Models",
        "figures/plots/big5_effect_size_models.png", figsize=(8, 5),
    )
    draw_demog_split(
        all_demog, n_demog_main, bonf_demog,
        title_suffix="Spearman Ï  â€”  6 Models",
        rho_outpath="figures/plots/demographic_correlation_models.png",
        r2_outpath="figures/plots/demographic_effect_size_models.png",
    )
    print(f"\nBonferroni thresholds:  Big5={bonf_big5:.5f}  Demographic={bonf_demog:.5f}")


run()
