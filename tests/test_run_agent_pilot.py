"""Tests for pilot mode, persona limiting, and the per-call diagnostics.

`--pilot` has to stay usable for configs whose output has been disabled, which
is the state every published config ends up in -- so its path resolution
deliberately does not go through `resolve_output_excel`.
"""

import types

import pytest
from openai import OpenAI

from scripts.pipeline.run_agent import (
    DEFAULT_PILOT_PERSONAS,
    EMPTY_META,
    get_model_response,
    resolve_output_excel,
    resolve_pilot_excel,
)


def norm(path):
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# pilot output path
# ---------------------------------------------------------------------------

def test_pilot_writes_beside_the_real_output():
    paths = {"output_excel": "results/model_size/Qwen3.5-9B_all_personas_results.xlsx"}
    assert norm(resolve_pilot_excel(paths, "Qwen3.5-9B")) == \
        "results/model_size/pilot/Qwen3.5-9B_pilot.xlsx"


def test_pilot_still_works_when_the_real_output_is_disabled():
    """A pilot cannot overwrite a completed run, so the guard must not block it."""
    paths = {"disabled_output_excel": "results/model_size/Qwen3.5-9B_all_personas_results.xlsx"}
    with pytest.raises(SystemExit):
        resolve_output_excel(paths, "configs/model_size/Qwen3.5-9B.json")
    assert norm(resolve_pilot_excel(paths, "Qwen3.5-9B")) == \
        "results/model_size/pilot/Qwen3.5-9B_pilot.xlsx"


def test_pilot_path_honours_the_historical_misspelling():
    paths = {"disabeled_output_excel": "results/X_all_personas_results.xlsx"}
    assert norm(resolve_pilot_excel(paths, "X")) == "results/pilot/X_pilot.xlsx"


def test_pilot_path_falls_back_to_results_when_nothing_is_declared():
    assert norm(resolve_pilot_excel({}, "X")) == "results/pilot/X_pilot.xlsx"


def test_pilot_never_collides_with_the_real_output():
    paths = {"output_excel": "results/model_size/M_all_personas_results.xlsx"}
    assert resolve_pilot_excel(paths, "M") != paths["output_excel"]


# ---------------------------------------------------------------------------
# persona limiting
# ---------------------------------------------------------------------------

def take(persona_names, limit):
    """The slice main() applies -- before the resume filter, so it is stable."""
    return persona_names[:limit] if limit is not None else persona_names


def test_limit_is_deterministic_across_runs():
    personas = sorted(f"P{i:04d}" for i in range(1, 400))
    assert take(personas, 5) == take(personas, 5)
    assert take(personas, 5) == ["P0001", "P0002", "P0003", "P0004", "P0005"]


def test_limit_larger_than_the_cohort_is_harmless():
    assert take(["P0001", "P0002"], 500) == ["P0001", "P0002"]


def test_default_pilot_size_is_small_enough_to_be_quick():
    assert 1 <= DEFAULT_PILOT_PERSONAS <= 10


# ---------------------------------------------------------------------------
# get_model_response diagnostics
# ---------------------------------------------------------------------------

def fake_openai(content, finish_reason="stop", prompt_tokens=2200,
                completion_tokens=900, reasoning_content=None):
    """A real OpenAI instance (so isinstance dispatch holds) with a stubbed call."""
    client = OpenAI(api_key="dummy", base_url="http://localhost:9/v1")
    message = types.SimpleNamespace(content=content, reasoning_content=reasoning_content)
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=types.SimpleNamespace(prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens),
    )
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client.chat.completions.create = create
    return client, captured


def test_usage_and_finish_reason_are_returned():
    client, _ = fake_openai('{"new_belief": 1}')
    content, meta = get_model_response(client, "m", "prompt", 16384, 0.7)
    assert content == '{"new_belief": 1}'
    assert meta["finish_reason"] == "stop"
    assert meta["prompt_tokens"] == 2200
    assert meta["completion_tokens"] == 900


def test_a_model_that_thinks_past_its_budget_returns_empty_content_not_none():
    """vLLM returns content=None on a truncated thought; parsing must not crash."""
    client, _ = fake_openai(None, finish_reason="length", completion_tokens=16384,
                            reasoning_content="thinking " * 500)
    content, meta = get_model_response(client, "m", "prompt", 16384, 0.7)
    assert content == ""
    assert meta["finish_reason"] == "length"
    assert meta["completion_tokens"] == 16384
    assert meta["reasoning_chars"] == len("thinking " * 500)


def test_chat_template_kwargs_are_forwarded_through_extra_body():
    """This is the only mechanism that turns thinking on for Gemma 4."""
    client, captured = fake_openai('{"new_belief": 1}')
    get_model_response(client, "m", "prompt", 1024, 0.7,
                       chat_template_kwargs={"enable_thinking": True})
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_extra_body_is_omitted_when_no_kwargs_are_configured():
    client, captured = fake_openai('{"new_belief": 1}')
    get_model_response(client, "m", "prompt", 1024, 0.7)
    assert captured["extra_body"] is None


def test_meta_keys_are_stable():
    """pilot_report reads these by name; run_agent stores them as columns."""
    client, _ = fake_openai("x")
    _, meta = get_model_response(client, "m", "p", 10, 0.0)
    assert set(meta) == set(EMPTY_META)


def test_unsupported_client_is_rejected():
    with pytest.raises(ValueError, match="Unsupported client"):
        get_model_response(object(), "m", "p", 10, 0.0)
