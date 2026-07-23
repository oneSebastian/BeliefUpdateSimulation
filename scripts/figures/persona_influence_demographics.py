"""
Plot 3: Demographic subgroup means â€” does the LLM respond differently
to personas of different ages, genders, and education levels?

For each demographic variable, compare mean Î” per subgroup across:
  - Human (ground truth)
  - Base condition (3 models)
  - No-demographic condition (3 models)

If demographic info influences LLM outputs, base subgroup means should
diverge; no-demographic should flatten them.

Layout: 3 rows (age / gender / education) Ã— 1 combined panel per row,
showing all models side by side.

Run from: HumanSimulationProject/
Output:   claude/persona_influence_demographics.png
"""

import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS


MODELS = {
    "GPT-5-mini": {
        "base":        "results/academic-ai-gpt-5-mini_all_personas_results.xlsx",
        "no-demog":    "results/ablations/academic-ai-gpt-5-mini_no-demographic.xlsx",
    },
    "Qwen3-32B": {
        "base":        "results/Qwen3_32B_all_personas_results.xlsx",
        "no-demog":    "results/ablations/Qwen3-32B_no-demographic.xlsx",
    },
    "Llama-3.3-70B": {
        "base":        "results/Llama-3.3-70B-Instruct_all_personas_results.xlsx",
        "no-demog":    "results/ablations/Llama-3.3-70B-Instruct_no-demographic.xlsx",
    },
}

MODEL_COLORS = {
    "GPT-5-mini":    {"base": "#c0392b", "no-demog": "#f1948a"},
    "Qwen3-32B":     {"base": "#d35400", "no-demog": "#f0b27a"},
    "Llama-3.3-70B": {"base": "#6c3483", "no-demog": "#c39bd3"},
}
with open("colors.yaml") as _f:
    HUMAN_COLOR = yaml.safe_load(_f)["human"]

GENDER_LABELS  = {0: "Female", 1: "Male", 2: "Diverse"}
QUAL_LABELS    = {
    1: "No formal",
    2: "Secondary",
    3: "Further ed.",
    4: "Bachelor's",
    5: "Master's",
    6: "PhD+",
}
# Collapse age into three bins
AGE_BINS   = [0, 35, 55, 100]
AGE_LABELS = ["Young\n(<35)", "Middle\n(35â€“55)", "Older\n(>55)"]


def normalize(df):
    df = df.copy()
    mask = df["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, "llm_new_belief"] *= -1
    df.loc[mask, "init_belief"]    *= -1
    return df


def extract_demog(demographic_str):
    d = json.loads(demographic_str)
    return pd.Series({
        "age":   d.get("age"),
        "gender_raw":  d.get("gender_raw"),
        "qual_raw":    d.get("highest_qualification_raw"),
    })


def load_llm_with_demog(path):
    df = pd.read_excel(path)
    df = normalize(df)
    df["delta"] = df["llm_new_belief"] - df["init_belief"]
    demog = df["demographic"].apply(extract_demog)
    df = pd.concat([df[["persona_id", "topic", "delta"]], demog], axis=1)
    df["age_bin"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    return df.dropna(subset=["delta"])


def load_human():
    df = pd.read_csv("results/merged_llm_participants_data_with_normalized_beliefs.csv")
    df["delta"]    = df["final_belief_normalized"] - df["initial_belief_normalized"]
    df["age_bin"]  = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df = df.rename(columns={"gender": "gender_raw",
                             "highest_qualification": "qual_raw"})
    return df[["persona_id", "topic", "delta", "age_bin", "gender_raw", "qual_raw"]]


def subgroup_means(df, col, categories):
    return df.groupby(col)["delta"].mean().reindex(categories)


def draw_subgroup_panel(ax, col, categories, cat_labels, human_df,
                        model_dfs, title):
    n      = len(categories)
    x      = np.arange(n)

    # number of series: human + 3Ã—(base + no-demog) = 7
    n_series = 1 + 2 * len(MODELS)
    total_w  = 0.8
    bw       = total_w / n_series
    positions = np.linspace(-(total_w - bw) / 2, (total_w - bw) / 2, n_series)

    # human
    h_means = subgroup_means(human_df, col, categories)
    ax.bar(x + positions[0], h_means, width=bw, color=HUMAN_COLOR,
           label="Human", edgecolor="white", zorder=2)

    si = 1
    for model_name, (base_df, nod_df) in model_dfs.items():
        for cond_key, df, label_suffix in [
            ("base",    base_df, "base"),
            ("no-demog", nod_df, "no-demog"),
        ]:
            means = subgroup_means(df, col, categories)
            color = MODEL_COLORS[model_name][cond_key]
            lbl   = f"{model_name} ({label_suffix})"
            ax.bar(x + positions[si], means, width=bw, color=color,
                   label=lbl, edgecolor="white", zorder=2)
            si += 1

    ax.axhline(0, color="grey", lw=0.7, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Î” Belief", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot():
    human_df = load_human()
    model_dfs = {
        model: (load_llm_with_demog(paths["base"]),
                load_llm_with_demog(paths["no-demog"]))
        for model, paths in MODELS.items()
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        "Demographic Subgroup Means: Do LLMs Respond Differently\n"
        "to Different Personas?  (Base vs No-Demographic)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # Row 1: age bins
    draw_subgroup_panel(
        axes[0], "age_bin", AGE_LABELS, AGE_LABELS,
        human_df, model_dfs, "Age Group",
    )

    # Row 2: gender (exclude code 2 â€” only 3 participants)
    gender_cats   = [0, 1]
    gender_labels = [GENDER_LABELS[g] for g in gender_cats]
    draw_subgroup_panel(
        axes[1], "gender_raw", gender_cats, gender_labels,
        human_df, model_dfs, "Gender",
    )

    # Row 3: education (all 6 levels)
    qual_cats   = sorted(QUAL_LABELS.keys())
    qual_labels = [QUAL_LABELS[q] for q in qual_cats]
    draw_subgroup_panel(
        axes[2], "qual_raw", qual_cats, qual_labels,
        human_df, model_dfs, "Highest Qualification",
    )

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig("figures/plots/persona_influence_demographics.png", dpi=150, bbox_inches="tight")
    plt.savefig("figures/plots/persona_influence_demographics.pdf", bbox_inches="tight")
    print("Saved figures/plots/persona_influence_demographics.png")


plot()
