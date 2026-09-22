"""Tests for the pilot diagnostics.

The pilot exists to answer two questions before 391 personas are committed to:
was the completion budget big enough, and did the model obey the JSON format?
Those look identical in the parsed columns -- an empty `llm_new_belief` either
way -- so the report separates them by `finish_reason`, and these tests pin
that separation.
"""

import pandas as pd
import pytest

from scripts.pipeline.pilot_report import (
    NEAR_LIMIT_FRACTION,
    load_workbook,
    summarize,
    verdict,
)


def frame(rows):
    """A pilot workbook with the diagnostic columns run_agent now writes."""
    defaults = dict(
        persona_id="P0001", topic="UBI", model="Test-9B",
        llm_new_belief=1, raw_response='{"new_belief": 1}',
        valid_output=True, n_tries=1, n_truncated=0, max_tokens_budget=16384,
        finish_reason="stop", prompt_tokens=2200, completion_tokens=900,
        completion_tokens_max=900, reasoning_chars=None,
        thinking_disabled=False, n_no_thinking=0,
    )
    return pd.DataFrame([{**defaults, **row} for row in rows])


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_clean_run_is_fully_valid():
    row = summarize(frame([{}, {}, {}]), "Test-9B")
    assert row["n"] == 3
    assert row["valid"] == 3
    assert row["valid_pct"] == 100
    assert row["truncated"] == 0
    assert verdict(row) == "OK"


def test_truncation_is_counted_from_the_finish_reason():
    df = frame([
        {},
        {"valid_output": False, "n_truncated": 1, "finish_reason": "length",
         "completion_tokens": 16384, "completion_tokens_max": 16384},
    ])
    row = summarize(df, "Test-9B")
    assert row["truncated"] == 1
    assert row["valid"] == 1
    assert "RAISE max_tokens" in verdict(row)


def test_truncation_recovered_by_a_retry_is_still_reported():
    """The bug this guards: recording only the last attempt hid the truncation.

    Attempt 1 ran out of budget, attempt 2 succeeded. The row is valid and its
    final finish_reason is 'stop', but the budget was still too small.
    """
    df = frame([
        {},
        {"valid_output": True, "n_tries": 2, "n_truncated": 1,
         "finish_reason": "stop", "completion_tokens": 900,
         "completion_tokens_max": 16384},
    ])
    row = summarize(df, "Test-27B")
    assert row["valid_pct"] == 100
    assert row["truncated"] == 1
    assert row["rows_truncated"] == 1
    assert "RAISE max_tokens" in verdict(row)
    assert "retries recovered" in verdict(row)


def test_token_high_water_mark_beats_the_final_attempt():
    """A recovered retry's small final count must not mask the large first one."""
    df = frame([{"completion_tokens": 900, "completion_tokens_max": 15000}])
    row = summarize(df, "m")
    assert row["tok_max"] == 15000


def test_multiple_truncated_attempts_on_one_row_are_all_counted():
    df = frame([{"valid_output": False, "n_tries": 3, "n_truncated": 3,
                 "finish_reason": "length", "completion_tokens_max": 16384}])
    row = summarize(df, "m")
    assert row["truncated"] == 3
    assert row["rows_truncated"] == 1


def test_format_failure_is_not_blamed_on_the_budget():
    """Unparseable output that stopped normally is a format problem."""
    df = frame([
        {},
        {"valid_output": False, "finish_reason": "stop", "completion_tokens": 120,
         "completion_tokens_max": 120,
         "raw_response": "Sure! The participant would probably agree."},
    ])
    row = summarize(df, "Test-0.8B")
    assert row["truncated"] == 0
    assert verdict(row).startswith("FORMAT")
    assert "budget was not the limit" in verdict(row)


def test_a_response_near_the_ceiling_is_flagged_before_it_truncates():
    """Nothing truncated, but the headroom is gone -- 391 personas will."""
    near = int(NEAR_LIMIT_FRACTION * 16384) + 10
    df = frame([
        {},
        {"completion_tokens": near, "completion_tokens_max": near},
    ])
    row = summarize(df, "Test-27B")
    assert row["truncated"] == 0
    assert row["near_limit"] == 1
    assert "RAISE max_tokens" in verdict(row)
    assert "OK" != verdict(row)


def test_budget_utilisation_is_reported_against_max_tokens():
    df = frame([{"completion_tokens": 8192, "completion_tokens_max": 8192,
                 "max_tokens_budget": 16384}])
    row = summarize(df, "Test-9B")
    assert row["budget"] == 16384
    assert row["tok_max"] == 8192
    assert row["budget_used_pct"] == pytest.approx(50.0)


def test_thinking_share_is_computed_from_reasoning_tokens():
    """vLLM reports reasoning_tokens inside usage.completion_tokens_details."""
    df = frame([{"completion_tokens": 1647, "reasoning_tokens": 1638,
                 "completion_tokens_max": 1647}])
    assert summarize(df, "m")["think_pct"] == pytest.approx(99.45, abs=0.1)


def test_thinking_share_is_absent_for_a_non_thinking_model():
    df = frame([{"reasoning_tokens": None}])
    assert pd.isna(summarize(df, "m")["think_pct"])


def test_thinking_share_ignores_rows_with_no_completion_count():
    df = frame([
        {"completion_tokens": 1000, "reasoning_tokens": 900},
        {"completion_tokens": None, "reasoning_tokens": None},
    ])
    assert summarize(df, "m")["think_pct"] == pytest.approx(90.0)


def test_rows_answered_without_thinking_are_counted():
    df = frame([
        {},
        {"n_tries": 3, "n_truncated": 2, "n_no_thinking": 1,
         "thinking_disabled": True, "completion_tokens_max": 16384},
    ])
    row = summarize(df, "Qwen3.5-2B")
    assert row["no_thinking"] == 1


def test_the_no_thinking_fallback_is_not_reported_as_a_budget_problem():
    """A row that only parsed once thinking was switched off is not evidence
    that the ceiling was too low -- the fallback exists because it was not."""
    df = frame([
        {},
        {"n_tries": 3, "n_truncated": 2, "n_no_thinking": 1,
         "thinking_disabled": True, "completion_tokens_max": 16384},
    ])
    row = summarize(df, "Qwen3.5-2B")
    assert "RAISE max_tokens" not in verdict(row)
    assert verdict(row).startswith("NO-THINKING FALLBACK")
    assert "1/2 row(s) answered with thinking off" in verdict(row)


def test_the_fallback_verdict_still_needs_attention():
    """Non-zero exit: the SLURM pilot job must fail visibly on it."""
    df = frame([{"n_tries": 3, "n_truncated": 2, "n_no_thinking": 1,
                 "thinking_disabled": True, "completion_tokens_max": 16384}])
    assert verdict(summarize(df, "m")) != "OK"


def test_truncation_without_the_fallback_still_recommends_a_bigger_budget():
    df = frame([{"valid_output": False, "n_truncated": 2, "n_tries": 2,
                 "finish_reason": "length", "completion_tokens_max": 16384}])
    assert "RAISE max_tokens" in verdict(summarize(df, "m"))


def test_no_thinking_defaults_to_zero_for_a_legacy_workbook():
    """Pilots run before the fallback existed have no such column."""
    df = pd.DataFrame([{"persona_id": "P0001", "model": "old", "llm_new_belief": 1}])
    assert summarize(df, "old")["no_thinking"] == 0


def test_retries_are_averaged():
    df = frame([{"n_tries": 1}, {"n_tries": 3}])
    assert summarize(df, "m")["mean_tries"] == pytest.approx(2.0)


def test_empty_workbook_reports_no_data():
    row = summarize(frame([]).iloc[0:0], "m")
    assert row["n"] == 0
    assert verdict(row).startswith("NO DATA")


def test_legacy_workbook_without_diagnostics_still_summarises():
    """Results written before the diagnostic columns existed must not crash."""
    df = pd.DataFrame([
        {"persona_id": "P0001", "topic": "UBI", "model": "old", "llm_new_belief": 1},
        {"persona_id": "P0002", "topic": "UBI", "model": "old", "llm_new_belief": None},
    ])
    row = summarize(df, "old")
    assert row["n"] == 2
    assert row["valid"] == 1
    assert row["tok_max"] is None


# ---------------------------------------------------------------------------
# round trip through a real workbook
# ---------------------------------------------------------------------------

def test_load_workbook_takes_the_label_from_the_model_column(tmp_path):
    path = tmp_path / "Test-9B_pilot.xlsx"
    frame([{}]).to_excel(path, index=False)
    label, df = load_workbook(path)
    assert label == "Test-9B"
    assert len(df) == 1


def test_load_workbook_falls_back_to_the_filename(tmp_path):
    path = tmp_path / "unnamed_pilot.xlsx"
    pd.DataFrame([{"persona_id": "P0001"}]).to_excel(path, index=False)
    label, _ = load_workbook(path)
    assert label == "unnamed_pilot"


def test_cli_exits_nonzero_when_a_model_needs_attention(tmp_path):
    from scripts.pipeline.pilot_report import main
    good = tmp_path / "good_pilot.xlsx"
    bad = tmp_path / "bad_pilot.xlsx"
    frame([{}]).to_excel(good, index=False)
    frame([{"valid_output": False, "n_truncated": 1, "finish_reason": "length",
            "completion_tokens": 16384,
            "completion_tokens_max": 16384}]).to_excel(bad, index=False)

    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1


def test_cli_reports_a_missing_file_rather_than_crashing(tmp_path):
    from scripts.pipeline.pilot_report import main
    assert main([str(tmp_path / "nope.xlsx")]) == 1
