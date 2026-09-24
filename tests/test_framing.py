"""Tests for the framing-effect analysis (scripts/stats/framing.py)."""

import pandas as pd
import pytest

from scripts.stats import framing as fr

POS = "Everyone should receive Universal Basic Income."
NEG = "There should be no Universal Basic Income."


def _row(pid, topic, formulation, init, final=0):
    row = {c: 0 for c in fr.HUMAN_COLS}
    row.update(persona_id=pid, topic=topic, statement_formulation=formulation,
               initial_belief=init, final_belief=final, second_order_belief=1,
               stance_message_1_shown=1, stance_message_2_shown=-1,
               stance_message_3_shown=0, rank_message_1_shown=3, familiarity=2)
    return row


@pytest.fixture
def csv_path(tmp_path):
    rows = [_row(1, "UBI", POS, 1, 2), _row(2, "UBI", NEG, -2, 0)]
    df = pd.DataFrame(rows)
    df["llm_new_belief_qwen"] = 99
    path = tmp_path / "human.csv"
    df.to_csv(path, index=False)
    return path


def test_raw_is_untouched(csv_path):
    raw, _ = fr.load_human_framing_data(csv_path)
    assert raw["initial_belief"].tolist() == [1, -2]
    assert raw["second_order_belief"].tolist() == [1, 1]
    assert raw["framing"].tolist() == ["pro", "anti"]
    assert raw["belief_change"].tolist() == [1, 2]
    assert "llm_new_belief_qwen" not in raw.columns


def test_normalized_flips_statement_frame_cols_only(csv_path):
    _, norm = fr.load_human_framing_data(csv_path)
    assert norm["initial_belief"].tolist() == [1, 2]
    assert norm["final_belief"].tolist() == [2, 0]
    assert norm["second_order_belief"].tolist() == [1, -1]
    assert norm["stance_message_2_shown"].tolist() == [-1, 1]
    assert norm["belief_change"].tolist() == [1, -2]
    assert norm["rank_message_1_shown"].tolist() == [3, 3]
    assert norm["familiarity"].tolist() == [2, 2]


def test_bonferroni_scales_and_caps():
    assert fr.bonferroni([0.01, 0.2, 0.5]) == pytest.approx([0.03, 0.6, 1.0])


def test_stars_thresholds():
    assert [fr.stars(p) for p in (0.0005, 0.005, 0.03, 0.05, 0.5)] == \
        ["***", "**", "*", "", ""]


def _normalized(pro_by_topic, anti_by_topic):
    rows = []
    for topic in pro_by_topic:
        for v in pro_by_topic[topic]:
            rows.append({"topic": topic, "framing": "pro", "initial_belief": v})
        for v in anti_by_topic[topic]:
            rows.append({"topic": topic, "framing": "anti", "initial_belief": v})
    return pd.DataFrame(rows)


def test_framing_tests_detects_shift_and_not_identity():
    same = [-2, -1, 0, 1, 2] * 20
    shifted_up = [0, 1, 1, 2, 2] * 20
    norm = _normalized({"a": same, "b": shifted_up}, {"a": same, "b": same})
    res = fr.framing_tests(norm).set_index("topic")

    assert res.loc["a", "chi2"] == pytest.approx(0.0)
    assert res.loc["a", "U_p"] == pytest.approx(1.0)
    assert res.loc["b", "chi2_p"] < 0.001
    assert res.loc["b", "U_p"] < 0.001
    assert res.loc["b", "U"] > res.loc["b", "U_null"]
    assert res.loc["b", "pro_counts"] == [0, 0, 20, 40, 40]
    assert res["chi2_p_bonf"].tolist() == pytest.approx(fr.bonferroni(res["chi2_p"].tolist()))


def _model_rows(human_raw, llm_new_belief):
    return pd.DataFrame({
        "persona_id": human_raw["persona_id"], "topic": human_raw["topic"],
        "statement_formulation": human_raw["statement_formulation"],
        "init_belief": human_raw["initial_belief"], "package": human_raw["package"],
        "llm_new_belief": llm_new_belief,
    })


def test_pairing_mismatches_counts_each_kind(csv_path):
    raw, _ = fr.load_human_framing_data(csv_path)
    model = _model_rows(raw, [0, 0])
    assert set(fr.pairing_mismatches(raw, model).values()) == {0}

    model.loc[0, "init_belief"] = 2
    model.loc[1, "statement_formulation"] = POS
    mm = fr.pairing_mismatches(raw, model)
    assert mm["init_belief"] == 1 and mm["formulation"] == 1
    assert fr.pairing_mismatches(raw, model.iloc[:1])["unmatched"] == 1


def test_model_deltas_normalizes_llm_against_human_init(csv_path):
    raw, norm = fr.load_human_framing_data(csv_path)
    # raw LLM answers on the worded scale: +1 on POS, +1 on NEG (= -1 normalized);
    # normalized human initial beliefs are [1, 2]
    deltas = fr.model_deltas(raw, norm, _model_rows(raw, [1, 1]))
    assert deltas["llm_final"].tolist() == [1, -1]
    assert deltas["delta"].tolist() == [1 - 1, -1 - 2]
    assert deltas["framing"].tolist() == ["pro", "anti"]


def test_model_deltas_drops_unparsed_and_rejects_mismatch(csv_path):
    raw, norm = fr.load_human_framing_data(csv_path)
    deltas = fr.model_deltas(raw, norm, _model_rows(raw, [1, float("nan")]))
    assert len(deltas) == 1

    bad = _model_rows(raw, [1, 1])
    bad.loc[0, "package"] = 1
    with pytest.raises(ValueError, match="package"):
        fr.model_deltas(raw, norm, bad)


def test_update_framing_tests_per_source_and_topic():
    same = [-1, 0, 0, 1] * 25
    up = [0, 1, 1, 2] * 25
    def frame(pro_by_topic, anti_by_topic):
        rows = [{"topic": t, "framing": f, "delta": v}
                for t in pro_by_topic
                for f, vals in (("pro", pro_by_topic[t]), ("anti", anti_by_topic[t]))
                for v in vals]
        return pd.DataFrame(rows)

    res = fr.update_framing_tests({
        "Human": frame({"a": same, "b": same}, {"a": same, "b": same}),
        "M": frame({"a": up, "b": same}, {"a": same, "b": same}),
    }).set_index(["source", "topic"])

    assert res.loc[("Human", "a"), "diff"] == pytest.approx(0.0)
    assert res.loc[("Human", "a"), "U_p"] == pytest.approx(1.0)
    assert res.loc[("M", "a"), "diff"] == pytest.approx(1.0)
    assert res.loc[("M", "a"), "U_p"] < 0.001
    # Bonferroni within source: m = 2 topics
    assert res.loc[("M", "a"), "U_p_bonf"] == pytest.approx(
        min(2 * res.loc[("M", "a"), "U_p"], 1.0))
    assert fr.render_updates(res.reset_index()).count("***") >= 1


def test_render_marks_significance_on_corrected_p():
    res = pd.DataFrame([{
        "topic": "t", "n_pro": 10, "n_anti": 10,
        "pro_counts": [2] * 5, "anti_counts": [2] * 5, "min_expected": 2.0,
        "chi2": 1.0, "dof": 4, "chi2_p": 0.02, "chi2_p_bonf": 0.06,
        "U": 60.0, "U_null": 50.0, "U_p": 0.001, "U_p_bonf": 0.003,
    }])
    text = fr.render(res)
    chi_line = next(l for l in text.splitlines() if l.startswith("t ") and "e-" in l and "60.0" not in l)
    u_line = next(l for l in text.splitlines() if l.startswith("t ") and "60.0" in l)
    assert "*" not in chi_line
    assert u_line.rstrip().endswith("**")
