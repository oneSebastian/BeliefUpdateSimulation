"""Tests for the retry loop and the no-thinking fallback.

The fallback exists because of what the Qwen3.5 pilots showed: every response
that ever parsed finished under ~12k tokens, while the ones that truncated ran
away past 19k and hit the 32k ceiling. There is no mass in between, so a third
*thinking* attempt on a row that has already truncated twice buys another full
budget of runaway. The third attempt therefore drops thinking instead.

Because that produces a row that is not comparable to the others, the two
things these tests pin hardest are: the switch reaches the server (it is sent
through `extra_body`, the only channel vLLM reads it from), and the row records
that it happened.
"""

import types

import pytest
from openai import OpenAI

from scripts.pipeline.run_agent import (
    TRUNCATIONS_BEFORE_DISABLING_THINKING,
    generate_with_retries,
    is_valid_output,
    parse_JSON_output,
)

GOOD = '{"new_belief": 1, "ranking": [1, 2, 3], "reasoning": "r", "general_public_stance": 0}'
JUNK = "Sure! The participant would probably agree."


def accept_json(text):
    """The real acceptance test used by the main (non-ablation) path."""
    return is_valid_output(*parse_JSON_output(text))


def scripted_client(attempts):
    """An OpenAI client that replays `attempts` in order.

    Each attempt is ``(content, finish_reason)``. Every call's `extra_body` is
    recorded, so a test can assert what was actually put on the wire rather
    than what the caller intended.
    """
    client = OpenAI(api_key="dummy", base_url="http://localhost:9/v1")
    sent = []
    remaining = list(attempts)

    def create(**kwargs):
        sent.append(kwargs.get("extra_body"))
        assert remaining, "the loop made more calls than the test scripted"
        content, finish_reason = remaining.pop(0)
        message = types.SimpleNamespace(content=content, reasoning_content=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message,
                                           finish_reason=finish_reason)],
            usage=types.SimpleNamespace(prompt_tokens=2200,
                                        completion_tokens=32768 if finish_reason == "length" else 900),
        )

    client.chat.completions.create = create
    return client, sent


def run(attempts, max_tries=3, chat_template_kwargs=None, accept=accept_json):
    client, sent = scripted_client(attempts)
    content, meta, stats = generate_with_retries(
        client, "m", "prompt",
        max_completion_tokens=32768,
        max_tries=max_tries,
        temperature=0.7,
        chat_template_kwargs=chat_template_kwargs,
        accept=accept,
    )
    return content, meta, stats, sent


THINKING = {"enable_thinking": True}


def thinking_flags(sent):
    """What `enable_thinking` each call actually carried."""
    return [None if body is None else body["chat_template_kwargs"]["enable_thinking"]
            for body in sent]


# ---------------------------------------------------------------------------
# the fallback itself
# ---------------------------------------------------------------------------

def test_two_truncations_disable_thinking_on_the_third_attempt():
    _, _, stats, sent = run(
        [(None, "length"), (None, "length"), (GOOD, "stop")],
        chat_template_kwargs=THINKING,
    )
    assert thinking_flags(sent) == [True, True, False]
    assert stats["n_tries"] == 3
    assert stats["n_truncated"] == 2
    assert stats["valid_output"] is True


def test_the_row_records_that_its_answer_came_from_the_fallback():
    """The whole point: a row answered without thinking is a different kind of
    observation, and the analysis has to be able to tell which ones they are."""
    _, _, stats, _ = run(
        [(None, "length"), (None, "length"), (GOOD, "stop")],
        chat_template_kwargs=THINKING,
    )
    assert stats["thinking_disabled"] is True
    assert stats["n_no_thinking"] == 1


def test_a_row_that_never_truncated_is_not_flagged():
    _, _, stats, sent = run([(GOOD, "stop")], chat_template_kwargs=THINKING)
    assert stats["thinking_disabled"] is False
    assert stats["n_no_thinking"] == 0
    assert thinking_flags(sent) == [True]


def test_a_row_recovered_before_the_threshold_keeps_thinking():
    _, _, stats, sent = run(
        [(None, "length"), (GOOD, "stop")],
        chat_template_kwargs=THINKING,
    )
    assert thinking_flags(sent) == [True, True]
    assert stats["thinking_disabled"] is False


def test_the_streak_resets_on_an_attempt_that_stopped_normally():
    """An isolated truncation either side of a format failure is not the
    runaway pattern the fallback is for, so thinking stays on."""
    _, _, stats, sent = run(
        [(None, "length"), (JUNK, "stop"), (None, "length")],
        chat_template_kwargs=THINKING,
    )
    assert thinking_flags(sent) == [True, True, True]
    assert stats["n_truncated"] == 2
    assert stats["n_no_thinking"] == 0
    assert stats["valid_output"] is False


def test_the_fallback_latches_once_a_no_thinking_attempt_stops_normally():
    """The bug this guards: a no-thinking attempt ends with finish_reason
    'stop' by design, so treating that as "the runaway broke" handed the next
    attempt back to thinking and straight back into the runaway."""
    _, _, stats, sent = run(
        [(None, "length"), (None, "length"), (JUNK, "stop"), (GOOD, "stop")],
        max_tries=4, chat_template_kwargs=THINKING,
    )
    assert thinking_flags(sent) == [True, True, False, False]


def test_the_fallback_persists_while_the_streak_does():
    """An attempt that truncates *without* thinking has not broken it either."""
    _, _, stats, sent = run(
        [(None, "length"), (None, "length"), (None, "length"), (GOOD, "stop")],
        max_tries=4, chat_template_kwargs=THINKING,
    )
    assert thinking_flags(sent) == [True, True, False, False]
    assert stats["n_no_thinking"] == 2
    assert stats["thinking_disabled"] is True
    assert stats["valid_output"] is True


def test_the_flag_describes_the_attempt_whose_answer_is_kept():
    """Attempts 1-2 truncate, so attempts 3 and 4 run without thinking. Both
    fail to parse, and the answer kept is attempt 4's -- made with thinking
    off, so the flag must say so."""
    _, _, stats, _ = run(
        [(None, "length"), (None, "length"), (JUNK, "stop"), (JUNK, "stop")],
        max_tries=4, chat_template_kwargs=THINKING,
    )
    assert stats["thinking_disabled"] is True
    assert stats["valid_output"] is False


def test_thinking_is_never_disabled_for_a_config_that_did_not_enable_it():
    """A plain API model has no chat template to switch, and sending
    `enable_thinking` to one is an error rather than a fallback."""
    _, _, stats, sent = run(
        [(None, "length"), (None, "length"), (JUNK, "stop")],
        chat_template_kwargs=None,
    )
    assert sent == [None, None, None]
    assert stats["n_no_thinking"] == 0
    assert stats["thinking_disabled"] is False


def test_other_chat_template_kwargs_survive_the_fallback():
    _, _, _, sent = run(
        [(None, "length"), (None, "length"), (GOOD, "stop")],
        chat_template_kwargs={"enable_thinking": True, "custom": "keep me"},
    )
    assert sent[-1]["chat_template_kwargs"] == {"enable_thinking": False,
                                                "custom": "keep me"}


def test_the_caller_s_kwargs_are_not_mutated():
    """`chat_template_kwargs` comes straight from the loaded config and is
    reused for every persona; flipping it in place would silently disable
    thinking for the rest of the run."""
    kwargs = {"enable_thinking": True}
    run([(None, "length"), (None, "length"), (GOOD, "stop")],
        chat_template_kwargs=kwargs)
    assert kwargs == {"enable_thinking": True}


def test_threshold_is_two_so_it_fits_inside_the_configured_max_tries():
    """The sweep configs all use max_tries = 3. A higher threshold would mean
    the fallback never fires."""
    assert TRUNCATIONS_BEFORE_DISABLING_THINKING == 2


# ---------------------------------------------------------------------------
# the surrounding accounting, which the fallback must not disturb
# ---------------------------------------------------------------------------

def test_the_successful_attempt_stops_the_loop():
    _, _, stats, sent = run([(GOOD, "stop"), (GOOD, "stop")], chat_template_kwargs=THINKING)
    assert stats["n_tries"] == 1
    assert len(sent) == 1


def test_truncations_are_counted_across_all_attempts():
    _, _, stats, _ = run([(None, "length"), (None, "length"), (GOOD, "stop")],
                         chat_template_kwargs=THINKING)
    assert stats["n_truncated"] == 2


def test_the_token_high_water_mark_survives_a_recovery():
    """The recovered attempt's small count must not hide the two that blew the
    budget -- that is what `completion_tokens_max` is for."""
    _, meta, stats, _ = run([(None, "length"), (None, "length"), (GOOD, "stop")],
                            chat_template_kwargs=THINKING)
    assert meta["completion_tokens"] == 900
    assert stats["completion_tokens_max"] == 32768


def test_exhausting_the_tries_returns_the_last_attempt():
    content, meta, stats, _ = run([(JUNK, "stop")] * 3, chat_template_kwargs=THINKING)
    assert content == JUNK
    assert stats["n_tries"] == 3
    assert stats["valid_output"] is False
    assert meta["finish_reason"] == "stop"


def test_a_truncated_attempt_with_no_content_does_not_crash_the_parser():
    content, _, stats, _ = run([(None, "length")] * 3, chat_template_kwargs=THINKING)
    assert content == ""
    assert stats["valid_output"] is False


def test_stats_keys_are_stable():
    """run_agent writes these as columns and pilot_report reads them by name."""
    _, _, stats, _ = run([(GOOD, "stop")])
    assert set(stats) == {"n_tries", "n_truncated", "n_no_thinking",
                          "thinking_disabled", "completion_tokens_max",
                          "valid_output"}


def test_a_custom_acceptance_test_is_honoured():
    """The probe-initial-belief ablation accepts a different shape entirely."""
    from scripts.pipeline.run_agent import _parse_initial_belief

    def accept(text):
        return _parse_initial_belief(text) in {-2, -1, 0, 1, 2}

    content, _, stats, _ = run([(JUNK, "stop"), ('{"belief": -1}', "stop")],
                               accept=accept)
    assert stats["valid_output"] is True
    assert _parse_initial_belief(content) == -1


# ---------------------------------------------------------------------------
# _parse_initial_belief
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"belief": 2}', 2),
    ('```json\n{"belief": -2}\n```', -2),
    ('<think>hmm</think>{"belief": 0}', 0),
    ("not json", None),
    ('{"other": 1}', None),
    ('{"belief": "x"}', None),
    ("", None),
])
def test_initial_belief_parsing(text, expected):
    from scripts.pipeline.run_agent import _parse_initial_belief
    assert _parse_initial_belief(text) == expected
