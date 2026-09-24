"""Statement framing (pro- vs anti-topic wording): effects on humans and LLMs.

Each participant saw every topic under one of two statement formulations (pro-
or anti-topic wording). All scale answers were given *relative to the statement
as worded* (see reference/otree_survey/survey/__init__.py), so:

- raw frame:        +2 = "agree strongly with the statement I was shown"
- normalized frame: +2 = "agree strongly with the pro-topic position"

Framing was randomly assigned per participant x topic, and the message packages
contain the same comments under both framings.

Part 1 -- framing effect on human initial belief
    Per topic, chi-squared test of homogeneity on the 2 (framing) x 5 (scale
    point) table, with a Mann-Whitney U test as backup. Initial belief is
    recorded before any message is shown.

Part 2 -- framing effect on the belief update, each source separately
    LLMs simulate the human with the same initial belief, so the framing effect
    on initial belief is shared. Per source (humans, each LLM) and topic, a
    Mann-Whitney U test of belief change (final - initial, normalized) under
    pro vs anti framing. Sources are compared by eye only, and the test is not
    conditioned on initial belief.

Bonferroni correction is applied within each test family (m = number of
topics); significance marks use the corrected thresholds.

    python -m scripts.stats.framing
"""

import argparse

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

from belief_update_sim.comment_ranks import MODELS
from belief_update_sim.config import (
    MERGED_HUMAN_CSV, RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs,
)
from belief_update_sim.normalization import statement_polarity

ALPHA = 0.05
LIKERT = [-2, -1, 0, 1, 2]
LABELS = ["SD", "D", "N", "A", "SA"]
RULE = "=" * 96
LINE = "-" * 96

# Scale answers the participant gave against the statement as worded. These are
# the columns that get flipped in the normalized frame.
STATEMENT_FRAME_COLS = [
    "initial_belief",
    "final_belief",
    "second_order_belief",
    "stance_message_1_shown",
    "stance_message_2_shown",
    "stance_message_3_shown",
]

HUMAN_COLS = [
    "persona_id", "topic", "topic_position", "package", "message_order_code",
    "message_1_shown", "message_2_shown", "message_3_shown",
    "statement_formulation",
    "initial_belief", "final_belief", "familiarity", "second_order_belief",
    "stance_message_1_shown", "stance_message_2_shown", "stance_message_3_shown",
    "rank_message_1_shown", "rank_message_2_shown", "rank_message_3_shown",
    "text_response",
    "age", "gender", "ethnicity", "country_of_birth", "country_of_residency",
    "student_status", "employment_status", "occupation_field",
    "highest_qualification",
    *[f"BIG_item_{i}" for i in range(1, 11)],
    *[f"BIG_item_{i}_label" for i in range(1, 11)],
    "BIG5_extraversion_score", "BIG5_agreeableness_score",
    "BIG5_conscientiousness_score", "BIG5_neuroticism_score",
    "BIG5_openness_score",
]

MODEL_COLS = ["persona_id", "topic", "statement_formulation", "init_belief",
              "package", "llm_new_belief"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_human_framing_data(path=MERGED_HUMAN_CSV):
    """Return (raw, normalized) human DataFrames, one row per participant x topic.

    raw        -- every human column exactly as recorded; no flips, no derived
                  columns. LLM columns and the CSV's own *_normalized /
                  *_belife_change / stance_direction columns are dropped.
    normalized -- same rows and columns, with STATEMENT_FRAME_COLS flipped into
                  the pro-topic frame.

    Both frames carry `polarity` (+1 pro wording, -1 anti wording), `framing`
    ("pro"/"anti") and `belief_change` (final - initial, in that frame's own
    coordinates), so either can be grouped by framing directly.
    """
    df = pd.read_csv(path)
    raw = df[HUMAN_COLS].copy()
    raw["polarity"] = raw["statement_formulation"].map(statement_polarity)
    raw["framing"] = raw["polarity"].map({1: "pro", -1: "anti"})

    normalized = raw.copy()
    normalized[STATEMENT_FRAME_COLS] = normalized[STATEMENT_FRAME_COLS].mul(
        normalized["polarity"], axis=0)

    for frame in (raw, normalized):
        frame["belief_change"] = frame["final_belief"] - frame["initial_belief"]

    # Sanity check against the normalization shipped in the merged CSV.
    if "initial_belief_normalized" in df.columns:
        assert (normalized["initial_belief"] == df["initial_belief_normalized"]).all()
        assert (normalized["final_belief"] == df["final_belief_normalized"]).all()

    return raw, normalized


def load_model_results(path):
    """The columns of a model results workbook needed for the update test."""
    df = pd.read_excel(path)[MODEL_COLS].copy()
    df["llm_new_belief"] = pd.to_numeric(df["llm_new_belief"], errors="coerce")
    return df


def pairing_mismatches(human_raw, model):
    """Count persona x topic rows where the LLM did not get the human's condition.

    Returns {check: count}; all zero means every LLM row saw the same statement
    formulation, initial belief and package as its human, and no row is missing
    on either side.
    """
    m = human_raw.merge(model, on=["persona_id", "topic"], how="outer",
                        suffixes=("_h", "_m"), indicator=True)
    both = m["_merge"] == "both"
    return {
        "unmatched": int((~both).sum()),
        "formulation": int((m.loc[both, "statement_formulation_h"]
                            != m.loc[both, "statement_formulation_m"]).sum()),
        "init_belief": int((m.loc[both, "initial_belief"]
                            != m.loc[both, "init_belief"]).sum()),
        "package": int((m.loc[both, "package_h"] != m.loc[both, "package_m"]).sum()),
    }


def model_deltas(human_raw, human_norm, model, name="model"):
    """One row per persona x topic with the LLM's belief change (normalized).

    The LLM starts from the human's initial belief, so delta = LLM final belief
    (flipped into the pro-topic frame) - human initial belief. Aborts if the LLM
    did not get exactly the human's condition. Rows whose LLM response did not
    parse are dropped.
    """
    bad = {k: v for k, v in pairing_mismatches(human_raw, model).items() if v}
    if bad:
        raise ValueError(f"{name}: LLM rows do not match the human condition: {bad}")

    h = human_norm[["persona_id", "topic", "framing", "polarity", "initial_belief"]]
    m = h.merge(model[["persona_id", "topic", "llm_new_belief"]],
                on=["persona_id", "topic"], validate="one_to_one")
    m = m.dropna(subset=["llm_new_belief"]).copy()
    m["llm_final"] = m["llm_new_belief"] * m["polarity"]
    m["delta"] = m["llm_final"] - m["initial_belief"]
    return m.drop(columns="llm_new_belief").reset_index(drop=True)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def bonferroni(p_values):
    """Bonferroni-adjusted p-values (p * m, capped at 1)."""
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < ALPHA:
        return "*"
    return ""


def framing_tests(normalized):
    """Per topic: chi-squared (framing x 5 scale points) and Mann-Whitney U on
    normalized initial belief, with Bonferroni-adjusted p-values per test family.
    Returns one row per topic."""
    rows = []
    for topic, g in normalized.groupby("topic"):
        pro = g.loc[g["framing"] == "pro", "initial_belief"]
        anti = g.loc[g["framing"] == "anti", "initial_belief"]
        table = pd.crosstab(g["framing"], g["initial_belief"]).reindex(
            index=["pro", "anti"], columns=LIKERT, fill_value=0)
        chi2, chi2_p, dof, expected = chi2_contingency(table)
        u, u_p = mannwhitneyu(pro, anti, alternative="two-sided")
        rows.append({
            "topic": topic, "n_pro": len(pro), "n_anti": len(anti),
            "pro_counts": table.loc["pro"].tolist(),
            "anti_counts": table.loc["anti"].tolist(),
            "min_expected": float(expected.min()),
            "chi2": float(chi2), "dof": int(dof), "chi2_p": float(chi2_p),
            "U": float(u), "U_null": len(pro) * len(anti) / 2, "U_p": float(u_p),
        })
    res = pd.DataFrame(rows)
    res["chi2_p_bonf"] = bonferroni(res["chi2_p"].tolist())
    res["U_p_bonf"] = bonferroni(res["U_p"].tolist())
    return res


def update_framing_tests(deltas_by_source):
    """Per source x topic: Mann-Whitney U on belief change (normalized), pro vs
    anti framing, with Bonferroni across topics within each source.

    deltas_by_source: {source: DataFrame with topic, framing, delta}.
    Returns one row per source x topic.
    """
    rows = []
    for source, df in deltas_by_source.items():
        block = []
        for topic, g in df.groupby("topic"):
            pro = g.loc[g["framing"] == "pro", "delta"]
            anti = g.loc[g["framing"] == "anti", "delta"]
            u, p = mannwhitneyu(pro, anti, alternative="two-sided")
            block.append({"source": source, "topic": topic,
                          "n_pro": len(pro), "n_anti": len(anti),
                          "mean_pro": pro.mean(), "mean_anti": anti.mean(),
                          "diff": pro.mean() - anti.mean(),
                          "U": float(u), "U_null": len(pro) * len(anti) / 2,
                          "U_p": float(p)})
        for r, p_bonf in zip(block, bonferroni([r["U_p"] for r in block])):
            r["U_p_bonf"] = p_bonf
        rows.extend(block)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(res):
    """Part 1: framing effect on human initial belief."""
    lines = []
    out = lines.append

    m = len(res)
    out(RULE)
    out("PART 1  Framing effect on human initial belief: pro- vs anti-worded statement")
    out("H0: the normalized initial-belief distribution is the same under both framings")
    out("H1: the distributions differ (two-sided)")
    out("initial belief on the normalized (pro-topic) 5-point scale, tested per topic")
    out(RULE)
    out("")

    header = "".join(f"{l:>6}" for l in LABELS)
    out(f"{'topic':<14}{'framing':<9}{header}{'total':>8}")
    out(LINE)
    for r in res.itertuples():
        for framing, counts, n in (("pro", r.pro_counts, r.n_pro),
                                   ("anti", r.anti_counts, r.n_anti)):
            out(f"{r.topic:<14}{framing:<9}" + "".join(f"{c:>6}" for c in counts)
                + f"{n:>8}")
    out("")

    out("Chi-squared test of homogeneity (2 x 5 table)")
    out(f"{'topic':<14}{'chi2':>10}{'dof':>5}{'p':>12}{'p_bonf':>12}{'':<5}{'min E':>8}")
    out(LINE)
    for r in res.itertuples():
        out(f"{r.topic:<14}{r.chi2:>10.3f}{r.dof:>5}{r.chi2_p:>12.3e}"
            f"{r.chi2_p_bonf:>12.3e} {stars(r.chi2_p_bonf):<4}{r.min_expected:>8.1f}")
    out("")

    out("Mann-Whitney U test (U for the pro group; U_null = n_pro * n_anti / 2)")
    out(f"{'topic':<14}{'U':>10}{'U_null':>10}{'p':>12}{'p_bonf':>12}")
    out(LINE)
    for r in res.itertuples():
        out(f"{r.topic:<14}{r.U:>10.1f}{r.U_null:>10.1f}{r.U_p:>12.3e}"
            f"{r.U_p_bonf:>12.3e} {stars(r.U_p_bonf)}")
    out(LINE)
    out(f"p_bonf: Bonferroni-adjusted within each test family (m = {m} topics)")
    out(f"significance on p_bonf: * < {ALPHA}, ** < 0.01, *** < 0.001")
    out("U > U_null: pro framing gives more pro-topic initial beliefs")
    out("min E: smallest expected cell count (chi-squared reliable when >= 5)")
    return "\n".join(lines) + "\n"


def render_updates(res):
    """Part 2: framing effect on the update, per source, compared by eye."""
    lines = []
    out = lines.append
    topics = list(dict.fromkeys(res["topic"]))
    m = len(topics)
    out(RULE)
    out("PART 2  Framing effect on the belief update, each source tested separately")
    out("delta = final - initial belief (normalized); Mann-Whitney U, pro vs anti, per topic")
    out("H0: belief change has the same distribution under both framings")
    out("H1: the distributions differ (two-sided)")
    out(RULE)
    out("")

    out(f"Summary: mean delta(pro) - mean delta(anti)  (raw p)  significance at "
        f"Bonferroni thresholds (m = {m})")
    out(f"{'source':<26}" + "".join(f"{t:>23}" for t in topics))
    out(LINE)
    for source, g in res.groupby("source", sort=False):
        g = g.set_index("topic")
        out(f"{source:<26}" + "".join(
            f"{g.loc[t, 'diff']:>+9.3f} ({g.loc[t, 'U_p']:.1e}) "
            f"{stars(g.loc[t, 'U_p_bonf']):<3}" for t in topics))
    out(f"stars: raw p < {ALPHA}/{m}, 0.01/{m}, 0.001/{m} "
        "(equivalent to p_bonf < 0.05, 0.01, 0.001)")
    out("")

    out("Details")
    out(f"{'source':<26}{'topic':<14}{'n_pro':>6}{'n_anti':>7}{'mean_pro':>10}"
        f"{'mean_anti':>10}{'U':>10}{'p':>12}{'p_bonf':>12}")
    out(LINE)
    for r in res.itertuples():
        out(f"{r.source:<26}{r.topic:<14}{r.n_pro:>6}{r.n_anti:>7}{r.mean_pro:>10.3f}"
            f"{r.mean_anti:>10.3f}{r.U:>10.1f}{r.U_p:>12.3e}{r.U_p_bonf:>12.3e} "
            f"{stars(r.U_p_bonf)}")
    out(LINE)
    out(f"p_bonf: Bonferroni-adjusted within each source (m = {m} topics)")
    out(f"significance on p_bonf: * < {ALPHA}, ** < 0.01, *** < 0.001")
    out("diff > 0: pro framing moves beliefs more towards the pro-topic side than "
        "anti framing does")
    out("sources are compared descriptively only; 'significant here, not there' is "
        "not itself\na test of a difference. Not conditioned on initial belief.")
    out("LLM responses that did not parse are excluded (visible as a lower n).")
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser().parse_args()
    raw, normalized = load_human_framing_data()

    deltas = {"Human": normalized.rename(columns={"belief_change": "delta"})}
    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        deltas[name] = model_deltas(raw, normalized, load_model_results(path), name)

    text = "\n".join([render(framing_tests(normalized)),
                      render_updates(update_framing_tests(deltas))])
    print(text)

    ensure_output_dirs()
    path = STATS_OUTPUT_DIR / "framing.txt"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
