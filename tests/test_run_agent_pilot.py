"""Tests for pilot mode, persona limiting, and the per-call diagnostics.

`--pilot` has to stay usable for configs whose output has been disabled, which
is the state every published config ends up in -- so its path resolution
deliberately does not go through `resolve_output_excel`.
"""

import json
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


def test_the_thinking_trace_is_kept_when_the_content_is_empty():
    """The whole point of a truncated row: 16k tokens were generated, and
    without a sample there is no way to tell looping from verbosity."""
    reasoning = "".join(f"step {i} of my reasoning. " for i in range(2000))
    client, _ = fake_openai(None, finish_reason="length", reasoning_content=reasoning)
    _, meta = get_model_response(client, "m", "p", 16384, 0.7)
    excerpt = meta["reasoning_excerpt"]
    assert excerpt, "thinking was discarded"
    assert excerpt.startswith("step 0 of my reasoning.")
    assert excerpt.rstrip().endswith("step 1999 of my reasoning.")
    assert "characters omitted" in excerpt


def test_the_excerpt_stays_inside_the_excel_cell_limit():
    """Excel refuses a cell over 32,767 characters; a 16k-token thought is ~60kB."""
    client, _ = fake_openai(None, reasoning_content="x" * 200_000)
    _, meta = get_model_response(client, "m", "p", 16384, 0.7)
    assert len(meta["reasoning_excerpt"]) < 32_767


def test_a_short_thought_is_kept_whole():
    client, _ = fake_openai('{"ok": 1}', reasoning_content="brief thought")
    _, meta = get_model_response(client, "m", "p", 1024, 0.7)
    assert meta["reasoning_excerpt"] == "brief thought"


# ---------------------------------------------------------------------------
# Real HTTP round-trip
#
# The bug these guard against slipped through because the tests above build the
# response by hand. A SimpleNamespace has whatever attribute you give it, so it
# cannot tell you what the server actually sends or what the SDK preserves.
# These serve vLLM's real wire format over a socket instead.
# ---------------------------------------------------------------------------

VLLM_030_RESPONSE = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
    "model": "Qwen3.5-2B",
    "choices": [{
        "index": 0, "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": '\n\n{"new_belief": 1}',
            "refusal": None, "annotations": None, "audio": None,
            "function_call": None,
            # vLLM 0.30 spells it "reasoning"; older versions used
            # "reasoning_content". Both must work.
            "reasoning": "Thinking Process:\n1. Analyze the persona...",
        },
    }],
    "usage": {
        "prompt_tokens": 21, "total_tokens": 1668, "completion_tokens": 1647,
        "completion_tokens_details": {"reasoning_tokens": 1638},
    },
}


@pytest.fixture
def vllm_server():
    """Serve one canned response over real HTTP; yields a base_url factory."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    state = {"body": VLLM_030_RESPONSE}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            payload = _json.dumps(state["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        yield state, OpenAI(api_key="dummy", base_url=f"http://127.0.0.1:{port}/v1")
    finally:
        server.shutdown()


def test_vllm_030_reasoning_field_is_captured_over_real_http(vllm_server):
    """The regression: vLLM 0.30 returns 'reasoning', not 'reasoning_content'.

    Reading only the latter discarded ~7.8k tokens per call in the first
    Qwen3.5 pilots while every other column looked healthy.
    """
    _, client = vllm_server
    content, meta = get_model_response(client, "Qwen3.5-2B", "p", 32768, 0.7)

    assert content == '\n\n{"new_belief": 1}'
    assert meta["reasoning_chars"] == len(VLLM_030_RESPONSE["choices"][0]["message"]["reasoning"])
    assert meta["reasoning_excerpt"].startswith("Thinking Process:")


def test_reasoning_token_count_is_read_from_usage_details(vllm_server):
    """More precise than a character count, and it is what max_tokens is
    compared against: completion_tokens includes the thinking."""
    _, client = vllm_server
    _, meta = get_model_response(client, "m", "p", 32768, 0.7)
    assert meta["completion_tokens"] == 1647
    assert meta["reasoning_tokens"] == 1638


def test_the_older_reasoning_content_spelling_still_works(vllm_server):
    """Older vLLM and other OpenAI-compatible servers use the other name."""
    state, client = vllm_server
    body = json.loads(json.dumps(VLLM_030_RESPONSE))
    body["choices"][0]["message"].pop("reasoning")
    body["choices"][0]["message"]["reasoning_content"] = "older spelling"
    state["body"] = body

    _, meta = get_model_response(client, "m", "p", 32768, 0.7)
    assert meta["reasoning_excerpt"] == "older spelling"


def test_a_server_that_reports_no_thinking_is_not_an_error(vllm_server):
    """Non-thinking models and servers without a parser must still work."""
    state, client = vllm_server
    body = json.loads(json.dumps(VLLM_030_RESPONSE))
    body["choices"][0]["message"].pop("reasoning")
    body["usage"].pop("completion_tokens_details")
    state["body"] = body

    content, meta = get_model_response(client, "m", "p", 32768, 0.7)
    assert content == '\n\n{"new_belief": 1}'
    assert meta["reasoning_tokens"] is None
    assert meta["reasoning_excerpt"] is None


def test_reasoning_content_is_found_when_the_sdk_hides_it_in_model_extra():
    """Extra fields are not part of the OpenAI schema; depending on the client
    version they arrive as an attribute or only inside model_extra. Reading
    just the attribute is what lost the trace on the first Qwen3.5-0.8B pilot."""
    client, _ = fake_openai(None, finish_reason="length")
    message = client.chat.completions.create().choices[0].message
    del message.reasoning_content
    message.model_extra = {"reasoning_content": "hidden thinking"}
    _, meta = get_model_response(client, "m", "p", 16384, 0.7)
    assert meta["reasoning_chars"] == len("hidden thinking")
    assert meta["reasoning_excerpt"] == "hidden thinking"


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
