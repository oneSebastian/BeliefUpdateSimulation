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


def _updates(p_by_cell, diff=-0.1):
    """An update_framing_tests result with chosen raw p-values.

    p_by_cell: {(source, topic): p}. Sources and topics keep insertion order.
    """
    return pd.DataFrame([{"source": s, "topic": t, "diff": diff, "U_p": p,
                          "U_p_bonf": 1.0}
                         for (s, t), p in p_by_cell.items()])


TOPICS = ("UBI", "penalty", "weight_loss")


def _six_models(p=0.5):
    """Human plus six LLMs over three topics: m = 21, threshold 0.05/21."""
    sources = [fr.HUMAN, *(f"M{i}" for i in range(6))]
    return {(s, t): p for s in sources for t in TOPICS}


def _cell(tex, source, topic_index):
    line = next(l for l in tex.splitlines() if l.startswith(source + " "))
    return line.split("&")[1 + topic_index].replace(r"\\", "").strip()


@pytest.mark.parametrize("x,expected", [
    (-0.078125, "-0.0781"), (0.1, "+0.1000"), (0.0, "+0.0000"),
    (-0.00001, "+0.0000"),
])
def test_tex_diff_has_four_places_and_no_negative_zero(x, expected):
    assert fr.tex_diff(x) == expected


@pytest.mark.parametrize("p,expected", [
    (0.6094, "$0.6094$"), (0.044491, "$0.0445$"), (0.00011, "$0.0001$"),
    (0.00004, "$<0.0001$"), (3.7e-22, "$<0.0001$"),
])
def test_tex_p_has_four_places_and_a_floor(p, expected):
    assert fr.tex_p(p) == expected


def test_tex_p_bold_stays_in_one_math_group():
    assert fr.tex_p(3.7e-22, bold=True) == r"$\mathbf{<0.0001}$"


def test_tex_bolds_below_bonferroni_over_all_tests_including_human():
    """0.05/21 = 0.00238: 0.0023 is bold, 0.0025 is not. Leaving the Human row
    out (0.05/18 = 0.00278) would bold 0.0025."""
    cells = _six_models()
    cells[("M0", "UBI")] = 0.0023
    cells[("M0", "penalty")] = 0.0025
    tex = fr.render_updates_tex(_updates(cells))
    assert _cell(tex, "M0", 0) == r"$-0.1000$ ($\mathbf{0.0023}$)"
    assert _cell(tex, "M0", 1) == r"$-0.1000$ ($0.0025$)"


def test_tex_holds_the_human_row_to_the_same_threshold():
    cells = _six_models()
    cells[(fr.HUMAN, "UBI")] = 0.0010
    tex = fr.render_updates_tex(_updates(cells))
    assert r"\mathbf{0.0010}" in _cell(tex, fr.HUMAN, 0)


def test_tex_reports_raw_not_adjusted_p():
    cells = _six_models()
    cells[("M1", "UBI")] = 0.0100
    tex = fr.render_updates_tex(_updates(cells))
    assert "0.0100" in _cell(tex, "M1", 0)
    assert "0.1800" not in tex and "1.0000" not in tex


def test_tex_bold_compares_the_unrounded_p():
    """0.002380 and 0.002382 both print as 0.0024, but sit either side of
    0.05/21 = 0.0023810."""
    cells = _six_models()
    cells[("M2", "UBI")] = 0.002380
    cells[("M2", "penalty")] = 0.002382
    tex = fr.render_updates_tex(_updates(cells))
    assert r"\mathbf" in _cell(tex, "M2", 0)
    assert r"\mathbf" not in _cell(tex, "M2", 1)


def test_tex_layout_matches_the_si_table():
    tex = fr.render_updates_tex(_updates(_six_models()))
    assert r"\label{tab:si_framing_updates}" in tex
    assert r"\begin{tabular}{l r r r}" in tex
    assert "Source & UBI & Penalty Shootout & Weight Loss Drugs \\\\" in tex
    lines = tex.splitlines()
    human = next(i for i, l in enumerate(lines) if l.startswith(fr.HUMAN + " "))
    assert lines[human - 1] == r"\midrule" and lines[human + 1] == r"\midrule"
    rows = [l for l in lines if l.endswith(r"\\") and not l.startswith("Source")]
    assert [l.split("&")[0].strip() for l in rows] == \
        [fr.HUMAN, *(f"M{i}" for i in range(6))]
    assert {l.count("&") for l in rows} == {3}


def test_tex_caption_states_raw_p_and_the_correction():
    caption = fr.render_updates_tex(_updates(_six_models())).split(r"\caption{")[1]
    assert "raw $p$-value" in caption
    assert "Bonferroni correction across the 21 tests" in caption
    assert r"3 topics $\times$ 7 sources" in caption
    assert "$p < 0.05/21$" in caption
    assert "^{*}" not in caption


def test_tex_refuses_a_table_without_llms():
    with pytest.raises(ValueError, match="no LLM"):
        fr.render_updates_tex(_updates({(fr.HUMAN, "UBI"): 0.5}))


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
