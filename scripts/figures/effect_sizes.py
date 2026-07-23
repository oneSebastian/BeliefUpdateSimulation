"""
Companion heatmaps to big5_correlation.png and demographic_correlation.png.
Shows ÏÂ² (% variance explained) instead of Ï.
Cell colour = magnitude of effect; sign shown in text; * = Bonferroni-corrected sig.

Run from: HumanSimulationProject/
Output:   claude/big5_effect_size.png
          claude/demographic_effect_size.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr, kruskal
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS

# â”€â”€ shared constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


MODELS = {
    "Qwen3-32B": {
        "Base":           "results/Qwen3_32B_all_personas_results.xlsx",
        "No persona":     "results/ablations/Qwen3-32B_no-persona.xlsx",
        "No personality": "results/ablations/Qwen3-32B_no-personality.xlsx",
        "No demographic": "results/ablations/Qwen3-32B_no-demographic.xlsx",
    },
    "Llama-3.3-70B": {
        "Base":           "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "No persona":     "results/ablations/Llama-3.3-70B-Instruct_no-persona.xlsx",
        "No personality": "results/ablations/Llama-3.3-70B-Instruct_no-personality.xlsx",
        "No demographic": "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx",
    },
    "GPT-5-mini": {
        "Base":           "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
        "No persona":     "results/ablations/academic-ai-gpt-5-mini_no-persona.xlsx",
        "No personality": "results/ablations/academic-ai-gpt-5-mini_no-personality.xlsx",
        "No demographic": "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx",
    },
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


def load_llm_big5(path, human_df):
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    big5_cols = list(BIG5.keys())
    persona_big5 = human_df[["persona_id"] + big5_cols].drop_duplicates("persona_id")
    df = df.merge(persona_big5, on="persona_id", how="inner")
    return df.dropna(subset=["delta"] + big5_cols)


def load_llm_demog(path):
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    def extract(s):
        d = json.loads(s)
        return pd.Series({
            "age":                   d.get("age"),
            "gender_raw":            d.get("gender_raw"),
            "highest_qualification": d.get("highest_qualification_raw"),
            "employment_status":     d.get("employment_status_raw"),
        })
    demog = df["demographic"].apply(extract)
    return pd.concat([df[["delta"]], demog], axis=1).dropna(subset=["delta"])


def rhos_ordinal(df, target="delta"):
    return rhos(df, ORDINAL_VARS, target)


def kw_nominal(df, col_map, target="delta"):
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


# â”€â”€ correlation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def rhos(df, col_map, target="delta"):
    out = {}
    for col, label in col_map.items():
        sub = df[[col, target]].dropna()
        if len(sub) < 10:
            out[label] = (np.nan, np.nan)
        else:
            r, p = spearmanr(sub[col], sub[target])
            out[label] = (r, p)
    return out


# â”€â”€ generic heatmap renderer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def draw_effect_heatmap(rows, col_labels, bonf_thr, init_row, title, outpath,
                        figsize, pad_right):
    """
    rows      : list of (row_label, group_label, rho_dict)
    init_row  : (row_label, group_label, rho_dict) â€” appended below separator
    bonf_thr  : Bonferroni p-threshold for main rows
    """
    all_rows = rows + [init_row]

    rho_mat  = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    pval_mat = np.array([[v[1] for v in r[2].values()] for r in all_rows])
    r2_mat   = rho_mat ** 2 * 100          # % variance explained
    r2_mat   = np.where(np.isnan(rho_mat), np.nan, r2_mat)

    n_rows, n_cols = rho_mat.shape
    vmax = 5.0   # cap colorscale at 5 % variance explained

    cmap = plt.cm.YlOrRd
    cmap.set_bad("lightgrey")

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(np.where(np.isnan(r2_mat), -1, r2_mat),
                   cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    # mask NaN cells manually (imshow doesn't support masked arrays well here)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(rho_mat[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             color="lightgrey", zorder=2))
                ax.text(j, i, "â€”", ha="center", va="center",
                        fontsize=8.5, zorder=3)
                continue

            rho  = rho_mat[i, j]
            r2   = r2_mat[i, j]
            pval = pval_mat[i, j]
            thr  = 0.05 if i == n_rows - 1 else bonf_thr
            sig  = "*" if pval < thr else ""

            # text: sign + rÂ² %
            sign = "+" if rho >= 0 else "âˆ’"
            txt  = f"{sign}{r2:.2f}%{sig}"

            # dark text on light cells, white on dark
            brightness = r2 / vmax
            color = "white" if brightness > 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color=color, fontweight="bold" if sig else "normal",
                    zorder=3)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([r[0] for r in all_rows], fontsize=9)

    # group separators + right-side labels
    prev_group = None
    drawn = set()
    for i, (_, group, _) in enumerate(all_rows):
        if group != prev_group and i > 0:
            lw = 3 if group == "Human init. belief" else 2
            ax.axhline(i - 0.5, color="white", lw=lw, zorder=4)
        prev_group = group
        if group not in drawn:
            indices = [j for j, (_, g, _) in enumerate(all_rows) if g == group]
            mid = (indices[0] + indices[-1]) / 2
            ax.annotate(group,
                        xy=(n_cols - 0.5, mid), xycoords="data",
                        xytext=(n_cols + pad_right, mid), textcoords="data",
                        fontsize=8.5, va="center", annotation_clip=False)
            drawn.add(group)

    cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.20 + pad_right * 0.035)
    cb.set_label("ÏÂ²  (% variance explained)", fontsize=9)
    cb.ax.set_title(f"0 â€“ {vmax:.0f}%", fontsize=8, pad=4)

    ax.set_title(
        title + f"\nÏÂ²  = variance explained  |  sign shows direction of Ï\n"
        f"* Bonferroni-corrected p < {bonf_thr:.4f}  (bottom row: p < 0.05 vs initial stance)",
        fontsize=9, pad=10,
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved {outpath}")


# â”€â”€ Big5 plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_big5():
    human_df = load_human()

    rows = [("Human", "Human", rhos(human_df, BIG5))]
    for model_name, conditions in MODELS.items():
        for cond_label, path in conditions.items():
            df = load_llm_big5(path, human_df)
            rows.append((cond_label, model_name, rhos(df, BIG5)))

    bonf_thr  = 0.05 / (len(rows) * len(BIG5))
    init_row  = ("Init. belief", "Human init. belief",
                 rhos(human_df, BIG5, target="initial_belief_normalized"))

    draw_effect_heatmap(
        rows       = rows,
        col_labels = list(BIG5.values()),
        bonf_thr   = bonf_thr,
        init_row   = init_row,
        title      = "Big5 Personality â†’ Belief Change (Î”):  Variance Explained (ÏÂ²)",
        outpath    = "figures/plots/big5_effect_size.png",
        figsize    = (9, 7),
        pad_right  = 0.3,
    )


# â”€â”€ Demographic plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fill_kw_effect_cells(ax, eps2_mat, pval_mat, n_main, bonf_thr, vmax):
    n_rows, n_cols = eps2_mat.shape
    for i in range(n_rows):
        for j in range(n_cols):
            eps2 = eps2_mat[i, j]
            pval = pval_mat[i, j]
            if np.isnan(eps2):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                             color="lightgrey", zorder=2))
                ax.text(j, i, "â€”", ha="center", va="center", fontsize=8.5, zorder=3)
                continue
            thr   = 0.05 if i == n_rows - 1 else bonf_thr
            sig   = "*" if pval < thr else ""
            txt   = f"{eps2:.3f}{sig}"
            color = "white" if eps2 / vmax > 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color=color, fontweight="bold" if sig else "normal", zorder=3)


def plot_demog():
    human_df = load_human()

    rows = [("Human", "Human",
             rhos(human_df, ORDINAL_VARS),
             kw_nominal(human_df, NOMINAL_VARS))]
    for model_name, conditions in MODELS.items():
        for cond_label, path in conditions.items():
            df = load_llm_demog(path)
            rows.append((cond_label, model_name,
                         rhos(df, ORDINAL_VARS),
                         kw_nominal(df, NOMINAL_VARS)))

    n_main   = len(rows)
    bonf_thr = 0.05 / (n_main * (len(ORDINAL_VARS) + len(NOMINAL_VARS)))

    init_row = (
        "Init. belief", "Human init. belief",
        rhos(human_df, ORDINAL_VARS, target="initial_belief_normalized"),
        kw_nominal(human_df, NOMINAL_VARS, target="initial_belief_normalized"),
    )
    all_rows = rows + [init_row]

    rho_mat = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    rho_p   = np.array([[v[1] for v in r[2].values()] for r in all_rows])
    kw_mat  = np.array([[v[0] for v in r[3].values()] for r in all_rows])
    kw_p    = np.array([[v[1] for v in r[3].values()] for r in all_rows])
    r2_mat  = rho_mat ** 2 * 100

    n_rows_n = len(all_rows)
    vmax_r2  = 5.0
    vmax_kw  = 0.05

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(9, 8),
        gridspec_kw={"width_ratios": [len(ORDINAL_VARS), len(NOMINAL_VARS)], "wspace": 0.55},
    )

    # Left: ÏÂ² for ordinal variables
    cmap_r2 = plt.cm.YlOrRd
    im_l = ax_l.imshow(np.where(np.isnan(r2_mat), -1, r2_mat),
                       cmap=cmap_r2, vmin=0, vmax=vmax_r2, aspect="auto")
    for i in range(n_rows_n):
        for j in range(len(ORDINAL_VARS)):
            rho  = rho_mat[i, j]
            r2   = r2_mat[i, j]
            pval = rho_p[i, j]
            if np.isnan(rho):
                ax_l.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, color="lightgrey", zorder=2))
                ax_l.text(j, i, "â€”", ha="center", va="center", fontsize=8.5, zorder=3)
                continue
            thr  = 0.05 if i == n_rows_n - 1 else bonf_thr
            sig  = "*" if pval < thr else ""
            sign = "+" if rho >= 0 else "âˆ’"
            txt  = f"{sign}{r2:.2f}%{sig}"
            color = "white" if r2 / vmax_r2 > 0.65 else "black"
            ax_l.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                      color=color, fontweight="bold" if sig else "normal", zorder=3)
    ax_l.set_xticks(range(len(ORDINAL_VARS)))
    ax_l.set_xticklabels(list(ORDINAL_VARS.values()), fontsize=10)
    ax_l.set_yticks(range(n_rows_n))
    ax_l.set_yticklabels([r[0] for r in all_rows], fontsize=9)
    prev_group = None
    for i, (_, group, *_) in enumerate(all_rows):
        if group != prev_group and i > 0:
            ax_l.axhline(i - 0.5, color="white", lw=3 if group == "Human init. belief" else 2)
        prev_group = group
    cb_l = plt.colorbar(im_l, ax=ax_l, shrink=0.5, pad=0.04)
    cb_l.set_label("ÏÂ²  (% variance explained)", fontsize=9)
    cb_l.ax.set_title(f"0â€“{vmax_r2:.0f}%", fontsize=8, pad=4)
    ax_l.set_title("Ordinal variables\n(ÏÂ²  â€” variance explained)", fontsize=9, pad=8)

    # Right: ÎµÂ² for nominal variables
    im_r = ax_r.imshow(np.where(np.isnan(kw_mat), -1, kw_mat),
                       cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax_kw, aspect="auto")
    _fill_kw_effect_cells(ax_r, kw_mat, kw_p, n_main, bonf_thr, vmax_kw)
    ax_r.set_xticks(range(len(NOMINAL_VARS)))
    ax_r.set_xticklabels(list(NOMINAL_VARS.values()), fontsize=10)
    ax_r.set_yticks(range(n_rows_n))
    ax_r.set_yticklabels([], fontsize=9)
    prev_group = None
    drawn = set()
    for i, (_, group, *_) in enumerate(all_rows):
        if group != prev_group and i > 0:
            ax_r.axhline(i - 0.5, color="white", lw=3 if group == "Human init. belief" else 2)
        prev_group = group
        if group not in drawn:
            indices = [j for j, (_, g, *_) in enumerate(all_rows) if g == group]
            mid = (indices[0] + indices[-1]) / 2
            ax_r.annotate(group,
                          xy=(len(NOMINAL_VARS) - 0.5, mid), xycoords="data",
                          xytext=(len(NOMINAL_VARS) + 0.25, mid), textcoords="data",
                          fontsize=8.5, va="center", annotation_clip=False)
            drawn.add(group)
    cb_r = plt.colorbar(im_r, ax=ax_r, shrink=0.5, pad=0.04)
    cb_r.set_label("KW ÎµÂ²  (variance explained)", fontsize=9)
    cb_r.ax.set_title(f"0â€“{vmax_kw:.0%}", fontsize=8, pad=4)
    ax_r.set_title("Nominal variables\n(Kruskal-Wallis ÎµÂ²)", fontsize=9, pad=8)

    fig.suptitle(
        "Demographic Attributes â†’ Belief Change (Î”):  Effect Sizes\n"
        f"sign = direction of Ï (ordinal only)   |   "
        f"* Bonferroni-corrected p < {bonf_thr:.4f}   (bottom row: p < 0.05 vs initial stance)",
        fontsize=9, y=1.01,
    )

    plt.savefig("figures/plots/demographic_effect_size.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/demographic_effect_size.pdf", bbox_inches="tight")
    print("Saved figures/plots/demographic_effect_size.png")


plot_big5()
plot_demog()
