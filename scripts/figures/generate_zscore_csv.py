"""
Generate all_models_combined_results.csv by running 7 mixed linear models.

For Llama, Qwen, GPT-5.2: belief shift columns already exist in the merged CSV.
For Claude, Gemini, GPT-5-mini: belief shifts are computed by joining the model
  xlsx files (results/*.xlsx) with the merged participant CSV.

Run from: HumanSimulationProjectFigures/
Output:   all_models_combined_results.csv
"""

import sys

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from belief_update_sim.normalization import NEGATIVE_FORMULATIONS
from belief_update_sim.config import DERIVED_DIR

DATA_FILE = "results/merged_llm_participants_data_with_normalized_beliefs.csv"


GENDER_LABELS     = {0: "Female", 1: "Male", 2: "Diverse"}
ETHNICITY_LABELS  = {0: "White", 1: "Asian", 2: "Black / African descent",
                     3: "Hispanic or Latino", 4: "Multiple ethnic groups",
                     5: "Middle Eastern / North African", 6: "Other ethnic group"}
EMPLOYMENT_LABELS = {0: "Employed full-time", 1: "Employed part-time", 2: "Self-employed",
                     3: "Unemployed", 4: "Not in labor force"}
EDUCATION_LABELS  = {1: "No formal education", 2: "Secondary school (GCSEs/O Levels)",
                     3: "Further education (A Levels, BTEC, apprenticeship)",
                     4: "Bachelor's degree", 5: "Master's degree", 6: "PhD or higher"}
COUNTRY_LABELS    = {'CN': 'China', 'DE': 'Germany', 'FR': 'France', 'HK': 'Hong Kong',
                     'IE': 'Ireland', 'LK': 'Sri Lanka', 'NG': 'Nigeria', 'PL': 'Poland',
                     'RU': 'Russia', 'GB': 'Great Britain'}
STUDENT_LABELS    = {1: "Full-time student", 2: "Part-time student",
                     3: "Not currently a student"}

DEMOGRAPHIC_VARS = ["gender", "ethnicity_grouped", "country_grouped", "student_grouped",
                    "employment_status", "highest_qualification"]
CONTINUOUS_VARS  = ["initial_belief_normalized", "age", "familiarity",
                    "BIG5_extraversion_score", "BIG5_agreeableness_score",
                    "BIG5_conscientiousness_score", "BIG5_neuroticism_score",
                    "BIG5_openness_score"]
REF_CATS = {
    "topic":                "UBI",
    "stance_direction":     "Pro",
    "gender":               "Female",
    "ethnicity_grouped":    "White",
    "country_grouped":      "Great Britain",
    "student_grouped":      "Not currently a student",
    "employment_status":    "Employed full-time",
    "highest_qualification":"Bachelor's degree",
}

# Models whose belief-shift column must be computed by joining the xlsx file.
# key = (description, xlsx_path)
XLSX_MODELS = [
    ("Claude Belief Shift",    "results/claude-opus-4-6_all_personas_results.xlsx"),
    ("Gemini Belief Shift",    "results/gemini-3-flash-preview_all_personas_results.xlsx"),
    ("GPT-5 Mini Belief Shift","results/academic-ai-gpt-5-mini_all_personas_results.xlsx"),
]


def load_base_data():
    df = pd.read_csv(DATA_FILE)

    for col, mapping in [("gender",               GENDER_LABELS),
                         ("ethnicity",             ETHNICITY_LABELS),
                         ("employment_status",     EMPLOYMENT_LABELS),
                         ("highest_qualification", EDUCATION_LABELS),
                         ("country_of_birth",      COUNTRY_LABELS),
                         ("student_status",        STUDENT_LABELS)]:
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    def get_topic(f):
        if not isinstance(f, str): return "Other"
        fl = f.lower()
        if "ubi" in fl or "universal basic income" in fl: return "UBI"
        if "penalty" in fl or "shootout" in fl:           return "Penalty"
        if "ozempic" in fl or "weight loss" in fl:        return "Weight Loss"
        return "Other"

    df["topic"] = df["statement_formulation"].apply(get_topic)

    # Belief shifts available in the merged CSV
    df["human_belief_change"]  = df["final_belief_normalized"]     - df["initial_belief_normalized"]
    df["norm_diff_final_llama"]= df["llm_new_belief_llama_normalized"] - df["initial_belief_normalized"]
    df["norm_diff_final_qwen"] = df["llm_new_belief_qwen_normalized"]  - df["initial_belief_normalized"]
    df["gpt_belief_shift"]     = df["llm_new_belief_gpt_normalized"]   - df["initial_belief_normalized"]

    # Collapse multi-level demographics to binary grouped variables
    df["ethnicity_grouped"] = df["ethnicity"].apply(
        lambda x: x if x == "White" else "Other"
    )
    df["student_grouped"] = df["student_status"].apply(
        lambda x: x if x == "Not currently a student" else "Other"
    )
    df["country_grouped"] = df["country_of_birth"].apply(
        lambda x: x if x == "Great Britain" else "Other"
    )

    for var in CONTINUOUS_VARS:
        if var in df.columns:
            df[f"{var}_centered"] = df[var] - df[var].mean()

    return df


def add_xlsx_model(base_df, description, xlsx_path):
    """
    Join model xlsx belief output with the base participant data.
    Returns base_df with a new target column added.
    """
    mdl = pd.read_excel(xlsx_path)

    # sign-flip for negative formulations (same as load_normalized_data)
    neg_mask = mdl["statement_formulation"].isin(NEGATIVE_FORMULATIONS)
    mdl.loc[neg_mask, "llm_new_belief"] *= -1

    # join on persona_id + statement_formulation (both are unique per run)
    mdl = mdl[["persona_id", "statement_formulation", "llm_new_belief"]].copy()
    mdl = mdl.rename(columns={"llm_new_belief": f"_new_belief_{description}"})

    merged = base_df.merge(mdl, on=["persona_id", "statement_formulation"], how="left")

    col = f"_new_belief_{description}"
    target_col = description.lower().replace(" ", "_").replace("-", "_")
    merged[target_col] = merged[col] - merged["initial_belief_normalized"]

    return merged, target_col


def run_model(df, target_var, description):
    model_cols = ([target_var, "topic", "stance_direction", "persona_id"]
                  + DEMOGRAPHIC_VARS
                  + [f"{v}_centered" for v in CONTINUOUS_VARS if f"{v}_centered" in df.columns])
    model_df = df[[c for c in model_cols if c in df.columns]].dropna()

    for col, ref in REF_CATS.items():
        if col not in model_df.columns:
            continue
        model_df[col] = model_df[col].astype("category").cat.remove_unused_categories()
        if ref in model_df[col].cat.categories:
            cats = model_df[col].cat.categories.tolist()
            model_df[col] = model_df[col].cat.reorder_categories(
                [ref] + [c for c in cats if c != ref])

    personality_vars = [f"{v}_centered" for v in CONTINUOUS_VARS
                        if "BIG5" in v and f"{v}_centered" in model_df.columns]
    predictors = (["C(topic)", "C(stance_direction)", "initial_belief_normalized_centered",
                   "age_centered", "familiarity_centered"]
                  + [f"C({v})" for v in DEMOGRAPHIC_VARS]
                  + personality_vars)

    final_predictors = []
    for p in predictors:
        col_name = p.replace("C(", "").replace(")", "")
        if col_name in model_df.columns and model_df[col_name].nunique() > 1:
            final_predictors.append(p)

    formula = f"{target_var} ~ {' + '.join(final_predictors)}"
    print(f"  n={len(model_df)}, participants={model_df['persona_id'].nunique()}")

    model = smf.mixedlm(formula, model_df, groups=model_df["persona_id"],
                        vc_formula={"topic": "0 + C(topic)"})
    try:
        result = model.fit(method="lbfgs", maxiter=1000)
    except Exception:
        result = model.fit(method="powell", maxiter=1000)

    rows = []
    for var in result.params.index:
        z = result.tvalues[var]
        p = result.pvalues[var]
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
        rows.append({
            "Model":        description,
            "Variable":     var,
            "Coefficient":  round(float(result.params[var]), 4),
            "Std_Error":    round(float(result.bse[var]),    4),
            "Z_value":      round(float(z),                  3),
            "P_value":      round(float(p),                  4),
            "Significance": sig,
            "Sample_Size":  len(model_df),
            "Participants": int(model_df["persona_id"].nunique()),
        })
    return rows


MODELS_CONFIG = [
    ("Human Belief Change",             "human_belief_change"),
    ("Human - Llama Bias (Difference)", "norm_diff_final_llama"),
    ("Human - Qwen Bias (Difference)",  "norm_diff_final_qwen"),
    ("GPT-5.2 Belief Shift",            "gpt_belief_shift"),
]


def main():
    print("Loading base data...")
    df = load_base_data()
    print(f"  {len(df)} rows loaded")

    # Add xlsx-derived models
    for description, xlsx_path in XLSX_MODELS:
        print(f"Joining {description} from {xlsx_path}...")
        df, target_col = add_xlsx_model(df, description, xlsx_path)
        MODELS_CONFIG.append((description, target_col))

    all_rows = []
    for description, target_var in MODELS_CONFIG:
        print(f"\nRunning: {description}")
        try:
            rows = run_model(df, target_var, description)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} coefficients extracted")
        except Exception as e:
            print(f"  ERROR: {e}")

    out = pd.DataFrame(all_rows)
    out.to_csv(f"{DERIVED_DIR}/all_models_combined_results.csv", index=False)
    print(f"\nSaved all_models_combined_results.csv ({len(out)} rows, {out['Model'].nunique()} models)")


if __name__ == "__main__":
    main()
