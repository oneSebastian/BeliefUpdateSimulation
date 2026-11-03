"""
ges.

Run from: HumanSimulationProject/
Input:    merged_llm_participants_data_with_normalized_beliefs_new.csv
Output:   all_models_combined_results.csv
"""

import sys
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

DATA = r"data/path"
OUT = Path("all_models_combined_results.csv")

# ── label maps (identical to the original notebook) ──────────────────────────
gender_labels = {0: "Female", 1: "Male", 2: "Diverse"}
ethnicity_labels = {0: "White", 1: "Asian", 2: "Black / African descent", 3: "Hispanic or Latino",
                    4: "Multiple ethnic groups", 5: "Middle Eastern / North African", 6: "Other ethnic group"}
employment_labels = {0: "Employed full-time", 1: "Employed part-time", 2: "Self-employed",
                     3: "Unemployed", 4: "Not in labor force"}
education_labels = {1: "No formal education", 2: "Secondary school (GCSEs/O Levels)",
                    3: "Further education (A Levels, BTEC, apprenticeship)", 4: "Bachelor's degree",
                    5: "Master's degree", 6: "PhD or higher"}
country_labels = {'CN': 'China', 'DE': 'Germany', 'FR': 'France', 'HK': 'Hong Kong', 'IE': 'Ireland',
                  'LK': 'Sri Lanka', 'NG': 'Nigeria', 'PL': 'Poland', 'RU': 'Russia', 'GB': 'Great Britain'}
student_labels = {1: "Full-time student", 2: "Part-time student", 3: "Not currently a student"}

df = pd.read_csv(DATA)
for col, m in [("gender", gender_labels), ("ethnicity", ethnicity_labels),
               ("employment_status", employment_labels), ("highest_qualification", education_labels),
               ("country_of_birth", country_labels), ("student_status", student_labels)]:
    df[col] = df[col].map(m).fillna(df[col])


def get_topic(f):
    f = str(f).lower()
    if "ubi" in f or "universal basic income" in f:
        return "UBI"
    if "penalty" in f or "shootout" in f:
        return "Penalty"
    if "ozempic" in f or "weight loss" in f:
        return "Weight Loss"
    return "Other"


df["topic"] = df["statement_formulation"].apply(get_topic)

# ── outcomes: (target column, model name used by the plotting script) ────────
df["human_belief_change"] = df["final_belief_normalized"] - df["initial_belief_normalized"]
df["norm_diff_final_llama"] = df["llm_new_belief_llama_normalized"] - df["initial_belief_normalized"]
df["norm_diff_final_qwen"] = df["llm_new_belief_qwen_normalized"] - df["initial_belief_normalized"]
df["gpt_belief_shift"] = df["llm_new_belief_gpt_normalized"] - df["initial_belief_normalized"]
df["claude_belief_shift"] = df["claude-opus-4-6_llm_new_belief_claude_normalized"] - df["initial_belief_normalized"]
df["gemini_belief_shift"] = df["gemini-3-flash-preview_llm_new_belief_gemini_normalized"] - df["initial_belief_normalized"]
df["gpt5_mini_belief_shift"] = df["academic-ai-gpt-5-mini_llm_new_belief_gpt5_normalized"] - df["initial_belief_normalized"]

MODELS = [
    ("norm_diff_final_llama",  "Human - Llama Bias (Difference)"),
    ("norm_diff_final_qwen",   "Human - Qwen Bias (Difference)"),
    ("claude_belief_shift",    "Claude Belief Shift"),
    ("gemini_belief_shift",    "Gemini Belief Shift"),
    ("human_belief_change",    "Human Belief Change"),
    ("gpt5_mini_belief_shift", "GPT-5 Mini Belief Shift"),
    ("gpt_belief_shift",       "GPT-5.2 Belief Shift"),
]

# ── centring ─────────────────────────────────────────────────────────────────
CONT = ["initial_belief_normalized", "age", "familiarity",
        "BIG5_extraversion_score", "BIG5_agreeableness_score", "BIG5_conscientiousness_score",
        "BIG5_neuroticism_score", "BIG5_openness_score"]
for v in CONT:
    df[f"{v}_centered"] = df[v] - df[v].mean()

# ── 80% dominance grouping (0 = modal category, 1 = other) ────────────────────
for src, dst in [("ethnicity", "ethnicity_grouped"), ("country_of_birth", "country_grouped"),
                 ("student_status", "student_grouped")]:
    counts = df[src].value_counts()
    assert counts.max() / counts.sum() >= 0.80, f"{src}: modal share < 80%"
    df[dst] = (df[src] != counts.idxmax()).astype(int)

CATS = {"topic": "UBI", "stance_direction": "Pro", "gender": "Female",
        "employment_status": "Employed full-time", "highest_qualification": "Bachelor's degree",
        "ethnicity_grouped": 0, "country_grouped": 0, "student_grouped": 0}
PREDICTORS = (["C(topic)", "C(stance_direction)", "initial_belief_normalized_centered",
               "age_centered", "familiarity_centered", "C(gender)", "C(employment_status)",
               "C(highest_qualification)", "C(ethnicity_grouped)", "C(country_grouped)",
               "C(student_grouped)"]
              + [f"{v}_centered" for v in CONT if v.startswith("BIG5")])


def fit_corrected(target):
    cols = [target, "persona_id"] + list(CATS) + [f"{v}_centered" for v in CONT]
    md = df[cols].dropna().copy()
    for c, ref in CATS.items():
        cats = sorted(md[c].unique(), key=lambda x: (x != ref, str(x)))   # reference first
        md[c] = pd.Categorical(md[c], categories=cats)
    formula = f"{target} ~ " + " + ".join(PREDICTORS)

    # corrected structure: participant random intercept, topic fixed only
    model = smf.mixedlm(formula, md, groups=md["persona_id"], re_formula="1")
    res, used = None, None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for method in ["lbfgs", "powell", "cg"]:
            try:
                res = model.fit(method=method, maxiter=2000, reml=True)
                used = method
                break
            except Exception:
                continue
    ols = smf.ols(formula, md).fit()
    if res is None:
        raise RuntimeError(f"{target}: mixed model could not be fitted")
    sigma_u = float(np.asarray(res.cov_re)[0, 0])
    same = np.allclose(res.fe_params.values, ols.params.values, atol=1e-3)
    msgs = sorted({str(x.message)[:70] for x in w if issubclass(x.category, (ConvergenceWarning, UserWarning))})
    print(f"  {target:24s} n={len(md)}  optimizer={used}  sigma_u^2={sigma_u:.2e}  "
          f"equals OLS={same}  warnings={msgs if msgs else 'none'}")
    fe = res.fe_params.index
    return pd.DataFrame({"Variable": fe, "Coefficient": res.fe_params.values,
                         "Std_Error": res.bse[fe].values, "Z_value": res.tvalues[fe].values,
                         "P_value": res.pvalues[fe].values})


rows = []
print("Fitting corrected models (participant random intercept, REML):")
for target, name in MODELS:
    r = fit_corrected(target)
    r.insert(0, "Model", name)
    rows.append(r)
out = pd.concat(rows, ignore_index=True)
out["Significance"] = np.select([out.P_value < .001, out.P_value < .01, out.P_value < .05],
                                ["***", "**", "*"], "")

if OUT.exists():
    shutil.copy(OUT, OUT.with_name("all_models_combined_results_OLD.csv"))
    print(f"Backed up previous file to {OUT.with_name('all_models_combined_results_OLD.csv')}")
out.round(6).to_csv(OUT, index=False)
print(f"Wrote {OUT} ({len(out)} rows)")

print("\nInitial-belief z-scores (corrected):")
print(out[out.Variable == "initial_belief_normalized_centered"][["Model", "Z_value"]]
      .round(2).to_string(index=False))
