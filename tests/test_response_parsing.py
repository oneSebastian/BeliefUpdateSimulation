import json

import pandas as pd

from belief_update_sim.response_parsing import parse_json_output, repair_dataframe

PAYLOAD = {
    "new_belief": 1,
    "ranking": [2, 1, 3],
    "reasoning": "Comment 2 was the most persuasive.",
    "general_public_stance": 0,
}


def test_parses_bare_json():
    assert parse_json_output(json.dumps(PAYLOAD))["new_belief"] == 1


def test_parses_response_with_only_a_closing_think_tag():
    """The Olmo-3 Think checkpoints emit reasoning then </think> then JSON,
    with no opening tag. run_agent.py's `<think>.*?</think>` misses this, which
    is why those runs exported as all-null."""
    raw = "Okay, let me work through this.\nStep one...\n</think>\n\n" + json.dumps(PAYLOAD)
    parsed = parse_json_output(raw)
    assert parsed is not None
    assert parsed["ranking"] == [2, 1, 3]


def test_parses_response_with_both_think_tags():
    raw = "<think>deliberating</think>\n" + json.dumps(PAYLOAD)
    assert parse_json_output(raw)["new_belief"] == 1


def test_coerces_string_valued_fields():
    """Think checkpoints emit "0" and ["2","1","3"] where Instruct models emit
    integers."""
    stringy = {"new_belief": "-2", "ranking": ["2", "1", "3"],
               "reasoning": "text", "general_public_stance": "1"}
    parsed = parse_json_output("</think>" + json.dumps(stringy))
    assert parsed["new_belief"] == -2
    assert parsed["ranking"] == [2, 1, 3]
    assert parsed["general_public_stance"] == 1


def test_returns_none_on_unparseable_input():
    assert parse_json_output("no json here at all") is None
    assert parse_json_output(None) is None
    assert parse_json_output(float("nan")) is None


def test_returns_none_on_truncated_json():
    """One row hit Excel's 32767-char cell cap and lost its trailing JSON."""
    assert parse_json_output("reasoning...\n</think>\n{\"new_belief\": 1, \"rank") is None


def test_returns_none_when_ranking_is_not_three_items():
    bad = dict(PAYLOAD, ranking=[1, 2])
    assert parse_json_output(json.dumps(bad)) is None


def test_returns_none_when_a_key_is_missing():
    bad = {k: v for k, v in PAYLOAD.items() if k != "general_public_stance"}
    assert parse_json_output(json.dumps(bad)) is None


def _frame(raw, **overrides):
    row = {"raw_response": raw, "llm_new_belief": None, "llm_general_public_stance": None,
           "llm_reasoning": None, "rank_1": None, "rank_2": None, "rank_3": None}
    row.update(overrides)
    return pd.DataFrame([row])


def test_repair_fills_null_columns():
    df = _frame("</think>" + json.dumps(PAYLOAD))
    assert repair_dataframe(df) == 1
    assert df.at[0, "llm_new_belief"] == 1
    assert df.at[0, "rank_2"] == 1
    assert df.at[0, "llm_reasoning"] == PAYLOAD["reasoning"]


def test_repair_never_overwrites_existing_values():
    df = _frame("</think>" + json.dumps(PAYLOAD), llm_new_belief=-2)
    repair_dataframe(df)
    assert df.at[0, "llm_new_belief"] == -2  # kept, not clobbered by the parsed 1


def test_repair_is_a_noop_when_nothing_is_null():
    df = _frame("</think>" + json.dumps(PAYLOAD), llm_new_belief=1,
                llm_general_public_stance=0, llm_reasoning="x",
                rank_1=2, rank_2=1, rank_3=3)
    assert repair_dataframe(df) == 0


def test_repair_leaves_unparseable_rows_null():
    df = _frame("garbage")
    assert repair_dataframe(df) == 0
    assert pd.isna(df.at[0, "llm_new_belief"])
