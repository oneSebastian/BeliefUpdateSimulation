"""
Demographic attributes â†’ Î” belief.
Ordinal variables (Age, Education): Spearman Ï heatmap.
Nominal variables (Gender, Employment): Kruskal-Wallis ÎµÂ² heatmap.
Both shown side-by-side in one figure.

Run from: HumanSimulationProject/
Output:   claude/demographic_correlation.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


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

ORDINAL_VARS = {
    "age":                   "Age",
    "highest_qualification": "Education",
}
NOMINAL_VARS = {
    "gender_raw":        "Gender",
    "employment_status": "Employment",
}


# â”€â”€ data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def normalize(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def extract_demog(s):
    d = json.loads(s)
    return pd.Series({
        "age":                   d.get("age"),
        "gender_raw":            d.get("gender_raw"),
        "highest_qualification": d.get("highest_qualification_raw"),
        "employment_status":     d.get("employment_status_raw"),
    })


def load_llm(path):
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    demog = df["demographic"].apply(extract_demog)
    return pd.concat([df[["delta"]], demog], axis=1).dropna(subset=["delta"])


def load_human():
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
    df = df.rename(columns={"gender": "gender_raw"})
    return df[["delta", "age", "gender_raw", "highest_qualification",
               "employment_status", "initial_belief_normalized"]]


# â”€â”€ correlation helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_rhos(df, col_map, target="delta"):
    out = {}
    for col, label in col_map.items():
        sub = df[[col, target]].dropna()
        if len(sub) < 10:
            out[label] = (np.nan, np.nan)
        else:
            rho, p = spearmanr(sub[col], sub[target])
            out[label] = (rho, p)
    return out


def compute_kw(df, col_map, target="delta"):
    """Kruskal-Wallis ÎµÂ² (epsilon-squared) for nominal variables."""
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
        eps2 = H / (N - 1)          # epsilon-squared: proportion of variance explained
        out[label] = (eps2, p)
    return out


# â”€â”€ heatmap drawing helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fill_rho_cells(ax, mat_rho, mat_p, n_main, bonf_thr, vmax=0.25):
    n_rows, n_cols = mat_rho.shape
    for i in range(n_rows):
        for j in range(n_cols):
            rho  = mat_rho[i, j]
            pval = mat_p[i, j]
            if np.isnan(rho):
                ax.text(j, i, "â€”", ha="center", va="center", fontsize=9)
                continue
            thr   = 0.05 if i == n_rows - 1 else bonf_thr
            sig   = "*" if pval < thr else ""
            txt   = f"{rho:+.2f}{sig}"
            color = "white" if abs(rho) / vmax > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if sig else "normal")


def _fill_kw_cells(ax, mat_eps2, mat_p, n_main, bonf_thr, vmax=0.05):
    n_rows, n_cols = mat_eps2.shape
    for i in range(n_rows):
        for j in range(n_cols):
            eps2 = mat_eps2[i, j]
            pval = mat_p[i, j]
            if np.isnan(eps2):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                             color="lightgrey", zorder=2))
                ax.text(j, i, "â€”", ha="center", va="center", fontsize=9, zorder=3)
                continue
            thr   = 0.05 if i == n_rows - 1 else bonf_thr
            sig   = "*" if pval < thr else ""
            txt   = f"{eps2:.3f}{sig}"
            color = "white" if eps2 / vmax > 0.65 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold" if sig else "normal", zorder=3)


def _draw_group_separators(ax, all_rows, n_main):
    prev_group = None
    for i, (_, group, *_) in enumerate(all_rows):
        if group != prev_group and i > 0:
            lw = 3 if group == "Human init. belief" else 2
            ax.axhline(i - 0.5, color="white", lw=lw)
        prev_group = group


def _draw_group_labels(ax, all_rows, n_cols, pad):
    drawn = set()
    for i, (_, group, *_) in enumerate(all_rows):
        if group not in drawn:
            indices = [j for j, (_, g, *_) in enumerate(all_rows) if g == group]
            mid = (indices[0] + indices[-1]) / 2
            ax.annotate(group,
                        xy=(n_cols - 0.5, mid), xycoords="data",
                        xytext=(n_cols + pad, mid), textcoords="data",
                        fontsize=8.5, va="center", annotation_clip=False)
            drawn.add(group)


# â”€â”€ main plot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot():
    human_df = load_human()

    rows = [("Human", "Human",
             compute_rhos(human_df, ORDINAL_VARS),
             compute_kw(human_df, NOMINAL_VARS))]
    for model_name, conditions in MODELS.items():
        for cond_label, path in conditions.items():
            df = load_llm(path)
            rows.append((cond_label, model_name,
                         compute_rhos(df, ORDINAL_VARS),
                         compute_kw(df, NOMINAL_VARS)))

    n_main   = len(rows)
    # Bonferroni across ALL main cells (ordinal + nominal combined)
    bonf_thr = 0.05 / (n_main * (len(ORDINAL_VARS) + len(NOMINAL_VARS)))

    init_row = (
        "Init. belief", "Human init. belief",
        compute_rhos(human_df, ORDINAL_VARS, target="initial_belief_normalized"),
        compute_kw(human_df, NOMINAL_VARS,   target="initial_belief_normalized"),
    )
    all_rows = rows + [init_row]

    # â”€â”€ build matrices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rho_mat  = np.array([[v[0] for v in r[2].values()] for r in all_rows])
    rho_p    = np.array([[v[1] for v in r[2].values()] for r in all_rows])
    kw_mat   = np.array([[v[0] for v in r[3].values()] for r in all_rows])
    kw_p     = np.array([[v[1] for v in r[3].values()] for r in all_rows])

    n_rows = len(all_rows)
    vmax_rho = 0.25
    vmax_kw  = 0.05

    # â”€â”€ figure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(9, 8),
        gridspec_kw={"width_ratios": [len(ORDINAL_VARS), len(NOMINAL_VARS)], "wspace": 0.55},
    )

    # Left: Spearman Ï (Age, Education)
    im_l = ax_l.imshow(rho_mat, cmap=plt.cm.RdBu, vmin=-vmax_rho, vmax=vmax_rho, aspect="auto")
    _fill_rho_cells(ax_l, rho_mat, rho_p, n_main, bonf_thr, vmax_rho)
    ax_l.set_xticks(range(len(ORDINAL_VARS)))
    ax_l.set_xticklabels(list(ORDINAL_VARS.values()), fontsize=11)
    ax_l.set_yticks(range(n_rows))
    ax_l.set_yticklabels([r[0] for r in all_rows], fontsize=9)
    _draw_group_separators(ax_l, all_rows, n_main)
    cb_l = plt.colorbar(im_l, ax=ax_l, shrink=0.5, pad=0.04)
    cb_l.set_label("Spearman Ï", fontsize=9)
    ax_l.set_title("Ordinal variables\n(Spearman Ï)", fontsize=9, pad=8)

    # Right: Kruskal-Wallis ÎµÂ² (Gender, Employment)
    kw_display = np.where(np.isnan(kw_mat), -1, kw_mat)
    im_r = ax_r.imshow(kw_display, cmap=plt.cm.YlOrRd, vmin=0, vmax=vmax_kw, aspect="auto")
    _fill_kw_cells(ax_r, kw_mat, kw_p, n_main, bonf_thr, vmax_kw)
    ax_r.set_xticks(range(len(NOMINAL_VARS)))
    ax_r.set_xticklabels(list(NOMINAL_VARS.values()), fontsize=11)
    ax_r.set_yticks(range(n_rows))
    ax_r.set_yticklabels([], fontsize=9)   # labels on left panel only
    _draw_group_separators(ax_r, all_rows, n_main)
    _draw_group_labels(ax_r, all_rows, len(NOMINAL_VARS), pad=0.25)
    cb_r = plt.colorbar(im_r, ax=ax_r, shrink=0.5, pad=0.04)
    cb_r.set_label("KW ÎµÂ²  (variance explained)", fontsize=9)
    cb_r.ax.set_title(f"0â€“{vmax_kw:.0%}", fontsize=8, pad=4)
    ax_r.set_title("Nominal variables\n(Kruskal-Wallis ÎµÂ²)", fontsize=9, pad=8)

    fig.suptitle(
        "Demographic Attributes â†’ Belief Change (Î”)\n"
        f"* Bonferroni-corrected p < {bonf_thr:.4f}   (bottom row: p < 0.05 vs initial stance)",
        fontsize=9, y=1.01,
    )

    plt.savefig("figures/plots/demographic_correlation.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/demographic_correlation.pdf", bbox_inches="tight")
    print(f"Saved figures/plots/demographic_correlation.png  (Bonferroni threshold: {bonf_thr:.4f})")


plot()
