"""Tests for the `no-reason` ablation.

The ablation repeats the main simulation with one change: the model is not
asked for a simulated reason for its belief update. The prompt must therefore
be the main template minus the reasoning field and its output rule -- nothing
else -- and an answer without a reasoning field must count as valid, or every
row would burn all its retries and be recorded as a failure.

Thinking is untouched: this ablation removes the written reason, not the
model's own reasoning.
"""

import json
import types
from pathlib import Path

import pytest
from openai import OpenAI

from scripts.pipeline.run_agent import (
    is_valid_output,
    json_example,
    json_example_no_reason,
    load_text,
    parse_JSON_output,
    process_persona,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

NO_REASON = '{"new_belief": 1, "ranking": [2, 1, 3], "general_public_stance": 0}'
WITH_REASON = ('{"new_belief": 1, "ranking": [2, 1, 3], "reasoning": "r", '
               '"general_public_stance": 0}')


@pytest.fixture
def at_repo_root(monkeypatch):
    """process_persona loads ablation templates by repo-relative path."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def persona_dir(tmp_path):
    """A synthetic persona with the same file layout as data/prolific_data."""
    d = tmp_path / "P9999"
    d.mkdir()
    demographic = {
        "participant_id": "P9999",
        "age": 40,
        "gender": "female",
        "country_of_residency": "United Kingdom",
        "personality": {f"survey.1.player.big{i}": 0 for i in range(1, 11)},
    }
    study = {
        "participant_id": "P9999",
        "topics": [{
            "topic": "UBI",
            "statement_formulation": "Everyone should receive Universal Basic Income.",
            "init_belief": -1,
            "familiarity": 1,
            "package": "package1",
            "message_order_index": 0,
            "message_order_perm": [1, 2, 3],
            "shown_messages": [
                {"shown_comment_id": i, "text": f"comment text {i}"} for i in (1, 2, 3)
            ],
        }],
    }
    (d / "demographic.json").write_text(json.dumps(demographic), encoding="utf-8")
    (d / "study_data.json").write_text(json.dumps(study), encoding="utf-8")
    return str(d)


def scripted_client(contents):
    """An OpenAI client that replays `contents` in order and records prompts."""
    client = OpenAI(api_key="dummy", base_url="http://localhost:9/v1")
    prompts = []
    remaining = list(contents)

    def create(**kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        assert remaining, "the loop made more calls than the test scripted"
        message = types.SimpleNamespace(content=remaining.pop(0), reasoning_content=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=2200, completion_tokens=100),
        )

    client.chat.completions.create = create
    return client, prompts


def run(persona_dir, contents, ablation, max_tries=3):
    client, prompts = scripted_client(contents)
    result = process_persona(
        client=client,
        persona_dir=persona_dir,
        template_text=load_text("prompt_templates/templates2.txt"),
        model="m",
        max_tokens_cfg=4096,
        max_tries_cfg=max_tries,
        temperature=0.7,
        ablation=ablation,
    )
    return result["results_by_topic"]["UBI"], prompts


# ---------------------------------------------------------------------------
# template
# ---------------------------------------------------------------------------

def test_template_differs_from_the_main_one_only_in_the_reasoning_rule(at_repo_root):
    main = load_text("prompt_templates/templates2.txt").splitlines()
    ablated = load_text("prompt_templates/no_reason.txt").splitlines()

    rules_start = main.index("OUTPUT RULES:")
    assert ablated[:rules_start + 1] == main[:rules_start + 1]

    strip_number = lambda line: line.split(") ", 1)[1]
    main_rules = [strip_number(r) for r in main[rules_start + 1:] if r.strip()]
    ablated_rules = [strip_number(r) for r in ablated[rules_start + 1:] if r.strip()]
    assert [r for r in main_rules if not r.startswith("reasoning")] == ablated_rules


def test_rules_are_renumbered_consecutively(at_repo_root):
    lines = load_text("prompt_templates/no_reason.txt").splitlines()
    rules = [r for r in lines[lines.index("OUTPUT RULES:") + 1:] if r.strip()]
    assert [r.split(")", 1)[0] for r in rules] == [str(i) for i in range(1, len(rules) + 1)]


def test_json_example_keeps_the_other_fields_in_order():
    def keys(example):
        return [line.split(":")[0].strip().strip('"')
                for line in example.splitlines() if ":" in line]
    assert keys(json_example_no_reason) == [k for k in keys(json_example) if k != "reasoning"]


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------

def test_missing_reasoning_is_valid_only_when_not_required():
    parsed = parse_JSON_output(NO_REASON)
    assert is_valid_output(*parsed, require_reasoning=False)
    assert not is_valid_output(*parsed)


def test_other_fields_are_still_checked_without_reasoning():
    bad = '{"new_belief": 5, "ranking": [2, 1, 3], "general_public_stance": 0}'
    assert not is_valid_output(*parse_JSON_output(bad), require_reasoning=False)


# ---------------------------------------------------------------------------
# process_persona
# ---------------------------------------------------------------------------

def test_prompt_asks_for_no_reasoning(at_repo_root, persona_dir):
    _, prompts = run(persona_dir, [NO_REASON], ablation="no-reason")
    assert "reasoning" not in prompts[0]
    assert json_example_no_reason in prompts[0]
    assert "comment text 2" in prompts[0]
    assert "init_belief = -1" in prompts[0]


def test_answer_without_reasoning_is_accepted_first_try(at_repo_root, persona_dir):
    row, prompts = run(persona_dir, [NO_REASON], ablation="no-reason")
    assert len(prompts) == 1
    assert row["valid_output"] is True
    assert row["n_tries"] == 1
    assert row["llm_new_belief"] == 1
    assert [row["rank_1"], row["rank_2"], row["rank_3"]] == [2, 1, 3]
    assert row["llm_general_public_stance"] == 0
    assert row["llm_reasoning"] is None


def test_thinking_trace_is_stripped_before_parsing(at_repo_root, persona_dir):
    """Qwen3-32B keeps thinking on and emits <think> inline in the content."""
    row, _ = run(persona_dir, ["<think>weighing it up</think>\n" + NO_REASON],
                 ablation="no-reason")
    assert row["valid_output"] is True
    assert row["llm_new_belief"] == 1


def test_main_run_still_rejects_an_answer_without_reasoning(at_repo_root, persona_dir):
    row, prompts = run(persona_dir, [NO_REASON] * 3, ablation=None)
    assert len(prompts) == 3
    assert row["valid_output"] is False
    assert "reasoning" in prompts[0]
