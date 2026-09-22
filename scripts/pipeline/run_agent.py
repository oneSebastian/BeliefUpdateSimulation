import os
import json
import re
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from xml.parsers.expat import model
from openpyxl.utils.exceptions import IllegalCharacterError
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# The API providers are optional. The cluster environment that serves the
# open-weight models (conda env `vllm`) has no reason to carry google-genai or
# anthropic, and importing them unconditionally made `run_agent` unimportable
# there. A missing provider is only an error once a config actually selects it.
try:
    from google import genai
    from google.genai.types import GenerateContentConfig, ThinkingConfig
except ImportError:  # pragma: no cover - depends on the installed extras
    genai = None

try:
    import anthropic
except ImportError:  # pragma: no cover - depends on the installed extras
    anthropic = None

from belief_update_sim.config import resolve_config_path
from scripts.pipeline.academic_ai import AcademicAIClient

# --------------------------------------------------
# Setup OpenAI client
# --------------------------------------------------
load_dotenv()  # loads OPENAI_API_KEY from .env


# --------------------------------------------------
# Utility loaders
# --------------------------------------------------
# utf-8-sig: the shipped configs and templates carry a UTF-8 BOM (they were
# written on Windows). Plain "utf-8" leaves it in the stream and json.load then
# fails on the very first character.
def load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


json_example = """{
  "new_belief": "INTEGER",
  "ranking": ["INTEGER", "INTEGER", "INTEGER"],
  "reasoning": "STRING",
  "general_public_stance": "INTEGER"
}"""

json_example_initial_belief = """{
  "belief": "INTEGER",
}"""

# --------------------------------------------------
# Prompt block builders (demographic.json + study_data.json)
# --------------------------------------------------
def build_demographic_block(demo: dict) -> str:
    """
    Feed ALL demographic fields (except personality and participant_id) to the model.
    NOTE: Does NOT include participant_id in the prompt.
    """
    keys = [k for k in demo.keys() if k not in {"personality", "participant_id"}]

    preferred = [
        "age_raw", "age",
        "gender_raw", "gender",
        "ethnicity_raw", "ethnicity",
        "country_of_residency_code", "country_of_residency",
        "country_of_origin_code", "country_of_origin",
        "student_status_raw", "student_status",
        "employment_status_raw", "employment_status",
        "occupation_field",
        "highest_qualification_raw", "highest_qualification",
    ]
    ordered = [k for k in preferred if k in keys] + [k for k in sorted(keys) if k not in preferred]

    lines = [f"{k}: {demo.get(k)}" for k in ordered]
    return "\n".join(lines) if lines else "No demographic data provided."


def build_personality_block(demo: dict) -> str:
    personality = demo.get("personality", {})
    if not personality:
        return "No personality data provided."

    def sort_key(item):
        k, _ = item
        m = re.search(r"big(\d+)$", k)
        return int(m.group(1)) if m else 999

    items = sorted(personality.items(), key=sort_key)
    return "\n".join([f"{k}: {v}" for k, v in items])


def _clean_json_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    quotes = "`“”‘’\"'"
    for quote in quotes:
        if cleaned.startswith(quote * 3) and cleaned.endswith(quote * 3):
            cleaned = cleaned[3:-3].strip()
        elif cleaned.startswith(quote) and cleaned.endswith(quote):
            cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace('“', '"').replace('”', '"')
    cleaned = cleaned.replace('‘', "'").replace('’', "'")
    if (cleaned.startswith("'''") and cleaned.endswith("'''")):
        cleaned = cleaned[3:-3].strip()
    elif cleaned.startswith('"""') and cleaned.endswith('"""'):
        cleaned = cleaned[3:-3].strip()
    elif (cleaned.startswith('"') and cleaned.endswith('"')) or \
         (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def build_comments_block(shown_messages: list) -> str:
    blocks = []
    for m in shown_messages:
        shown_id = m.get("shown_comment_id")
        text = (m.get("text") or "").strip()
        blocks.append(f"Comment {shown_id}:\n{text}")
    return "\n\n".join(blocks)


# --------------------------------------------------
# Parse LLM output: numeric belief + ranking + general public stance (numeric)
# --------------------------------------------------
def parse_numeric_output(response_text: str):
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]

    new_belief: Optional[int] = None
    ranking: List[int] = []
    general_public: Optional[int] = None

    for line in lines:
        if re.fullmatch(r"-2|-1|0|1|2", line):
            new_belief = int(line)
            break

    for line in lines:
        if line.upper().startswith("RANKING"):
            ids = re.findall(r"\d+", line)
            ranking = [int(x) for x in ids]
            break

    for line in lines:
        if line.upper().startswith("GENERAL_PUBLIC_STANCE"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                candidate = parts[1].strip()
                if re.fullmatch(r"-2|-1|0|1|2", candidate):
                    general_public = int(candidate)
            break

    return new_belief, ranking, general_public, response_text


def parse_JSON_output(response_text: str):
    new_belief: Optional[int] = None
    ranking: List[int] = []
    reasoning: Optional[str] = None
    general_public: Optional[int] = None

    cleaned_text = _clean_json_text(response_text)

    # Parse JSON
    try:
        data = json.loads(cleaned_text)

        # Parse new_belief
        if isinstance(data.get("new_belief"), int):
            new_belief = data["new_belief"]

        # Parse ranking
        if isinstance(data.get("ranking"), list):
            ranking = [
                int(x) for x in data["ranking"]
                if isinstance(x, int)
            ]
        
        # Parse reasoning
        if isinstance(data.get("reasoning"), str):
            reasoning = data["reasoning"]

        # Parse general_public_stance
        if isinstance(data.get("general_public_stance"), int):
            general_public = data["general_public_stance"]

    except json.JSONDecodeError:
        # Leave values as None / empty if JSON parsing fails
        pass

    return new_belief, ranking, reasoning, general_public


def is_valid_output(
    new_belief: Optional[int],
    ranking: List[int],
    reasoning: Optional[str],
    general_public: Optional[int]
) -> bool:
    if new_belief not in {-2, -1, 0, 1, 2}:
        return False
    if general_public not in {-2, -1, 0, 1, 2}:
        return False
    if sorted(ranking) != [1, 2, 3]:
        return False
    if not isinstance(reasoning, str):
        return False
    return True


def _is_instance_of(client, module, attribute) -> bool:
    """isinstance() against a provider that may not be installed."""
    if module is None:
        return False
    return isinstance(client, getattr(module, attribute))


# What a call cost and why it stopped. `finish_reason == "length"` is the one
# that matters for the model-size sweep: a thinking model that exhausts its
# completion budget mid-thought returns no JSON at all, which is indistinguishable
# from a formatting failure unless the stop reason is recorded alongside it.
EMPTY_META = {
    "finish_reason": None,
    "prompt_tokens": None,
    # vLLM counts thinking tokens in completion_tokens, so this is the number
    # to compare against max_tokens when sizing the budget.
    "completion_tokens": None,
    # How much of completion_tokens was thinking, as reported by the server.
    "reasoning_tokens": None,
    "reasoning_chars": None,
    "reasoning_excerpt": None,
}

# A 32k-token thought is ~120kB, and Excel refuses a cell over 32,767 characters,
# so the thinking is stored as a head-and-tail sample rather than in full. Head
# and tail together are what distinguish a model looping on one phrase from one
# reasoning at length and running out of room: a loop shows the same text at
# both ends.
REASONING_EXCERPT_CHARS = 1500


def _excerpt(text: str, limit: int = REASONING_EXCERPT_CHARS) -> str:
    if len(text) <= 2 * limit:
        return text
    omitted = len(text) - 2 * limit
    return f"{text[:limit]}\n\n...[{omitted} characters omitted]...\n\n{text[-limit:]}"


# Servers disagree on the name. vLLM 0.30 returns "reasoning"; earlier vLLM and
# several other OpenAI-compatible servers return "reasoning_content". Reading
# only one of them silently discards the entire thinking trace -- which is what
# happened to the first Qwen3.5 pilots: ~7.8k tokens per call generated and
# recorded nowhere.
REASONING_FIELDS = ("reasoning", "reasoning_content")


def _extra_field(obj, field):
    """Read a non-schema field however the SDK exposes it."""
    value = getattr(obj, field, None)
    if value:
        return value
    for holder in ("model_extra", "__pydantic_extra__"):
        extra = getattr(obj, holder, None)
        if isinstance(extra, dict) and extra.get(field):
            return extra[field]
    return None


def _reasoning_text(message):
    for field in REASONING_FIELDS:
        value = _extra_field(message, field)
        if value:
            return value
    return None


def _reasoning_tokens(usage):
    """Exact thinking-token count, when the server reports one.

    vLLM puts it in usage.completion_tokens_details.reasoning_tokens. It is the
    number that matters for sizing max_tokens: completion_tokens includes the
    thinking, and this says how much of it was.
    """
    details = getattr(usage, "completion_tokens_details", None)
    if details is None:
        details = _extra_field(usage, "completion_tokens_details")
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get("reasoning_tokens")
    return getattr(details, "reasoning_tokens", None)


# How many budget-exhausted attempts in a row it takes before the next attempt
# gives up on thinking. Two is the point at which the pilots stop being
# ambiguous: across the Qwen3.5 pilots every response that ever parsed finished
# under 12k tokens, while the ones that truncated ran away past 19k and hit the
# 32k ceiling. A third *thinking* attempt on such a row buys another full budget
# of runaway; a third attempt without thinking at least has a chance of
# answering. The fallback is recorded per row -- see `thinking_disabled` in the
# diagnostics -- because an answer produced without thinking is not comparable
# to one produced with it.
TRUNCATIONS_BEFORE_DISABLING_THINKING = 2


def _parse_initial_belief(text: str) -> Optional[int]:
    """The `probe-initial-belief` ablation's single output value, or None."""
    try:
        return int(json.loads(_clean_json_text(text))["belief"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def get_model_response(
    client,
    model: str,
    prompt: str,
    max_completion_tokens: int | None,
    temperature: float,
    chat_template_kwargs: Optional[dict] = None,
):
    """Return ``(content, meta)``.

    ``content`` is always a string -- a model that spends its whole budget
    thinking returns ``None`` for the message content, and the callers parse it.
    """
    meta = dict(EMPTY_META)

    if _is_instance_of(client, genai, "Client"):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_completion_tokens,
                thinking_config=ThinkingConfig(thinking_level="low")
            ),
        )
        content = response.text
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            meta["finish_reason"] = str(getattr(candidates[0], "finish_reason", None))
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_token_count", None)
            meta["completion_tokens"] = getattr(usage, "candidates_token_count", None)
    elif _is_instance_of(client, anthropic, "Anthropic"):
        response = client.messages.create(
            model=model,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        content_blocks = response.content
        content = "".join(block.text for block in content_blocks
                          if getattr(block, "type", None) == "text")
        meta["finish_reason"] = getattr(response, "stop_reason", None)
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "input_tokens", None)
            meta["completion_tokens"] = getattr(usage, "output_tokens", None)
    elif isinstance(client, OpenAI):
        # vLLM takes `chat_template_kwargs` (enable_thinking) through extra_body;
        # the OpenAI API itself ignores an absent one, so only send it when set.
        extra_body = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else None
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            extra_body=extra_body,
        )
        choice = response.choices[0]
        content = choice.message.content
        meta["finish_reason"] = choice.finish_reason
        usage = getattr(response, "usage", None)
        if usage is not None:
            meta["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            meta["completion_tokens"] = getattr(usage, "completion_tokens", None)
            meta["reasoning_tokens"] = _reasoning_tokens(usage)
        # With a reasoning parser the thinking is split off into its own field
        # and is NOT part of `content`, so an empty `content` alongside a long
        # thinking trace is the signature of a model that thought itself out of
        # its budget. Keep a sample: without it a truncated run records that 32k
        # tokens were generated but nothing about what they said, which is
        # exactly when you need to know.
        reasoning = _reasoning_text(choice.message)
        if reasoning:
            meta["reasoning_chars"] = len(reasoning)
            meta["reasoning_excerpt"] = _excerpt(reasoning)
    elif isinstance(client, AcademicAIClient):
        response = client.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            maxTokens=max_completion_tokens,
        )
        content = response["data"]["content"]
    else:
        raise ValueError("Unsupported client type")

    return (content if isinstance(content, str) else ""), meta


def generate_with_retries(
    client,
    model: str,
    prompt: str,
    max_completion_tokens: int | None,
    max_tries: int,
    temperature: float,
    chat_template_kwargs: Optional[dict],
    accept,
):
    """Call the model until ``accept(content)`` holds, at most ``max_tries`` times.

    Returns ``(content, meta, stats)`` for the *last* attempt made -- the one
    that succeeded, or the last that failed. ``stats`` accumulates across
    attempts, because a retry that recovers still tells you what the first
    attempt cost.

    After :data:`TRUNCATIONS_BEFORE_DISABLING_THINKING` consecutive *thinking*
    attempts end with ``finish_reason == "length"``, thinking is switched off
    for the rest of this row's attempts. A thinking attempt that stops normally
    resets the streak -- an isolated truncation is not the runaway pattern this
    guards against -- but a no-thinking attempt stopping normally does not,
    because that is simply what no-thinking attempts do. Without the latch,
    one such attempt would hand the next one back to thinking and straight
    back into the runaway.
    """
    stats = {
        "n_tries": 0,
        "n_truncated": 0,
        # How many attempts ran without thinking, and whether the attempt whose
        # answer is being kept was one of them.
        "n_no_thinking": 0,
        "thinking_disabled": False,
        "completion_tokens_max": None,
        "valid_output": False,
    }
    content, meta = "", dict(EMPTY_META)

    # Only meaningful when the config asked for thinking in the first place.
    # For a plain API model there is no chat template to switch, and sending
    # `enable_thinking` to one would be an error rather than a fallback.
    thinking_configurable = bool((chat_template_kwargs or {}).get("enable_thinking"))
    truncated_streak = 0
    disabled = False

    for _ in range(max_tries):
        stats["n_tries"] += 1

        if thinking_configurable and truncated_streak >= TRUNCATIONS_BEFORE_DISABLING_THINKING:
            disabled = True  # latches for the rest of this row
        stats["thinking_disabled"] = disabled
        if disabled:
            stats["n_no_thinking"] += 1
            attempt_kwargs = {**chat_template_kwargs, "enable_thinking": False}
        else:
            attempt_kwargs = chat_template_kwargs

        content, meta = get_model_response(
            client, model, prompt,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            chat_template_kwargs=attempt_kwargs,
        )

        if meta["finish_reason"] == "length":
            stats["n_truncated"] += 1
            truncated_streak += 1
        elif not disabled:
            truncated_streak = 0
        if meta["completion_tokens"] is not None:
            stats["completion_tokens_max"] = (
                meta["completion_tokens"] if stats["completion_tokens_max"] is None
                else max(stats["completion_tokens_max"], meta["completion_tokens"])
            )

        if accept(content):
            stats["valid_output"] = True
            break

    return content, meta, stats


# --------------------------------------------------
# Process a single persona
# --------------------------------------------------
def process_persona(
    client: Union[OpenAI, AcademicAIClient, None],
    persona_dir: str,
    template_text: str,
    model: str,
    max_tokens_cfg: int | None,
    max_tries_cfg: int = 1,
    temperature: float = 0.0,
    single_topic: Optional[str] = None,
    print_prompt: bool = False,
    ablation: Optional[str] = None,
    results_path: Optional[str] = None,
    chat_template_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
    demographic_path = os.path.join(persona_dir, "demographic.json")
    study_path = os.path.join(persona_dir, "study_data.json")

    if not os.path.exists(demographic_path) or not os.path.exists(study_path):
        print(f"Skipping {persona_dir}: missing demographic.json or study_data.json")
        return {}

    demographic = load_json(demographic_path)
    study = load_json(study_path)

    persona_id = os.path.basename(persona_dir.rstrip("/\\"))
    topics = study.get("topics", [])
    if not topics:
        print(f"Skipping {persona_dir}: no topics in study_data.json")
        return {}

    demographic_block = build_demographic_block(demographic)
    personality_block = build_personality_block(demographic)

    results_by_topic: Dict[str, Any] = {}

    for t in topics:
        topic = str(t.get("topic") or "")
        if not topic:
            continue
        if single_topic and topic != single_topic:
            continue

        statement_formulation = (t.get("statement_formulation") or "").strip()
        init_belief = t.get("init_belief", None)
        familiarity = t.get("familiarity", None)
        shown_messages = t.get("shown_messages", [])

        if not statement_formulation or init_belief is None:
            print(f"WARNING: missing statement/init_belief for {persona_id} {topic}. Skipping.")
            continue
        if familiarity is None:
            familiarity = "Not provided"

        comments_block = build_comments_block(shown_messages)

        user_prompt = template_text.format(
            demographic_block=demographic_block,
            personality_block=personality_block,
            topic=topic,
            statement_formulation=statement_formulation,
            init_belief=init_belief,
            familiarity=familiarity,
            comments_block=comments_block,
            json_example=json_example,
        )

        # perform ablation change
        if ablation is None:
            pass
        elif ablation == "all-positive":
            template_text = load_text("prompt_templates/templates2.txt")
            with open("prompt_templates/topic_map.json", "r") as f:
                topic_map = json.load(f)
            assert statement_formulation in topic_map, f"Statement formulation '{statement_formulation}' not found in topic_map.json"
            user_prompt = template_text.format(
                demographic_block=demographic_block,
                personality_block=personality_block,
                topic=topic,
                statement_formulation=topic_map[statement_formulation][0],
                init_belief=init_belief,
                familiarity=familiarity,
                comments_block=comments_block,
                json_example=json_example,
            )
        elif ablation == "all-negative":
            template_text = load_text("prompt_templates/templates2.txt")
            with open("prompt_templates/topic_map.json", "r") as f:
                topic_map = json.load(f)
            assert statement_formulation in topic_map, f"Statement formulation '{statement_formulation}' not found in topic_map.json"
            user_prompt = template_text.format(
                demographic_block=demographic_block,
                personality_block=personality_block,
                topic=topic,
                statement_formulation=topic_map[statement_formulation][1],
                init_belief=init_belief,
                familiarity=familiarity,
                comments_block=comments_block,
                json_example=json_example,
            )
        elif ablation == "no-demographic":
            template_text = load_text("prompt_templates/no_demographic.txt")
            user_prompt = template_text.format(
                personality_block=personality_block,
                topic=topic,
                statement_formulation=statement_formulation,
                init_belief=init_belief,
                familiarity=familiarity,
                comments_block=comments_block,
                json_example=json_example,
            )
        elif ablation == "no-personality":
            template_text = load_text("prompt_templates/no_personality.txt")
            user_prompt = template_text.format(
                demographic_block=demographic_block,
                topic=topic,
                statement_formulation=statement_formulation,
                init_belief=init_belief,
                familiarity=familiarity,
                comments_block=comments_block,
                json_example=json_example,
            )
        elif ablation == "no-persona":
            template_text = load_text("prompt_templates/no_persona.txt")
            user_prompt = template_text.format(
                topic=topic,
                statement_formulation=statement_formulation,
                init_belief=init_belief,
                comments_block=comments_block,
                json_example=json_example,
            )
        elif ablation == "probe-initial-belief":
            template_text = load_text("prompt_templates/probe_initial_belief.txt")
            user_prompt = template_text.format(
                demographic_block=demographic_block,
                personality_block=personality_block,
                topic=topic,
                statement_formulation=statement_formulation,
                familiarity=familiarity,
                json_example=json_example_initial_belief,
            )
        elif ablation == "own-initial-belief":
            initial_belief_path = results_path.replace("own-initial-belief", "probe-initial-belief")
            assert Path(initial_belief_path).exists(), f"Initial belief results file not found: {initial_belief_path}"
            df = pd.read_excel(initial_belief_path)
            mask = (df["persona_id"] == persona_id) & (df["topic"] == topic)
            assert mask.sum() == 1, "Expected exactly one matching row"
            persona_row = df.loc[
                (df["persona_id"] == persona_id) &
                (df["topic"] == topic)
            ].iloc[0]
            init_belief = persona_row["initial_belief"]
            user_prompt = template_text.format(
                demographic_block=demographic_block,
                personality_block=personality_block,
                topic=topic,
                statement_formulation=statement_formulation,
                init_belief=init_belief,
                familiarity=familiarity,
                comments_block=comments_block,
                json_example=json_example,
            )

        else:
            raise NotImplementedError(f"{ablation} has not yet been implemented as an ablation setting.")

        if print_prompt:
            print("\n========== PROMPT SENT TO MODEL ==========\n")
            print(user_prompt)
            print("\n==========================================\n")

        probing_initial_belief = ablation in ["probe-initial-belief"]
        if probing_initial_belief:
            def accept(text):
                return _parse_initial_belief(text) in {-2, -1, 0, 1, 2}
        else:
            def accept(text):
                return is_valid_output(*parse_JSON_output(text))

        content, meta, stats = generate_with_retries(
            client, model, user_prompt,
            max_completion_tokens=max_tokens_cfg,
            max_tries=max_tries_cfg,
            temperature=temperature,
            chat_template_kwargs=chat_template_kwargs,
            accept=accept,
        )

        if probing_initial_belief:
            initial_belief = _parse_initial_belief(content)
        else:
            llm_new_belief, ranking, reasoning, llm_general_public = parse_JSON_output(content)

        # Why the last attempt ended, what it cost, and how many it took. For
        # small or heavily-thinking models these are the difference between
        # "the model cannot follow the format" and "the token budget was too
        # small", which look identical in the parsed columns alone.
        # `thinking_disabled` says whether the answer in this row was produced
        # with thinking switched off by the fallback, which makes it a different
        # kind of observation from the rest and must be visible downstream.
        diagnostics = {
            "n_tries": stats["n_tries"],
            "n_truncated": stats["n_truncated"],
            "n_no_thinking": stats["n_no_thinking"],
            "thinking_disabled": stats["thinking_disabled"],
            "valid_output": stats["valid_output"],
            "max_tokens_budget": max_tokens_cfg,
            "finish_reason": meta["finish_reason"],
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": meta["completion_tokens"],
            "completion_tokens_max": stats["completion_tokens_max"],
            "reasoning_tokens": meta["reasoning_tokens"],
            "reasoning_chars": meta["reasoning_chars"],
            "reasoning_excerpt": meta["reasoning_excerpt"],
        }

        if probing_initial_belief:
            results_by_topic[topic] = {
                # identifiers
                "persona_id": persona_id,
                "topic": topic,
                "model": model,

                "statement_formulation": statement_formulation,
                "familiarity": familiarity,
                "demographic": demographic,
                "prompt_sent": user_prompt,

                # outputs
                "initial_belief": initial_belief,
                "raw_response": content,
                **diagnostics,
            }
        else:
            rank_1 = ranking[0] if len(ranking) > 0 else None
            rank_2 = ranking[1] if len(ranking) > 1 else None
            rank_3 = ranking[2] if len(ranking) > 2 else None

            results_by_topic[topic] = {
                # identifiers
                "persona_id": persona_id,
                "topic": topic,
                "model": model,

                # inputs (exact)
                "statement_formulation": statement_formulation,
                "init_belief": init_belief,
                "familiarity": familiarity,
                "package": t.get("package"),
                "message_order_index": t.get("message_order_index"),
                "message_order_perm": t.get("message_order_perm"),
                "shown_messages": shown_messages,
                "demographic": demographic,
                "prompt_sent": user_prompt,

                # outputs
                "llm_new_belief": llm_new_belief,
                "rank_1": rank_1,
                "rank_2": rank_2,
                "rank_3": rank_3,
                "llm_general_public_stance": llm_general_public,
                "llm_reasoning": reasoning,
                "raw_response": content,
                **diagnostics,
            }

        if single_topic:
            break

    return {"persona_id": persona_id, "results_by_topic": results_by_topic}


# --------------------------------------------------
# Incremental Excel writer (checkpointing)
# --------------------------------------------------
def append_rows_to_excel(
    output_excel: str,
    new_rows: List[Dict[str, Any]],
    dedupe_keys: List[str],
):
    os.makedirs(os.path.dirname(output_excel) or ".", exist_ok=True)

    new_df = pd.DataFrame(new_rows)

    # stringify complex columns for Excel
    for col in ["demographic", "shown_messages", "message_order_perm", "prompt_sent", "raw_response"]:
        if col in new_df.columns:
            new_df[col] = new_df[col].apply(lambda x: safe_json_dumps(x) if isinstance(x, (dict, list)) else x)

    if os.path.exists(output_excel):
        old_df = pd.read_excel(output_excel)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Keep last occurrence per key (latest run wins)
    if all(k in combined.columns for k in dedupe_keys):
        combined["_row_order"] = range(len(combined))
        combined = combined.sort_values("_row_order").drop_duplicates(subset=dedupe_keys, keep="last")
        combined = combined.drop(columns=["_row_order"])
    try:
        combined.to_excel(output_excel, index=False)
    except IllegalCharacterError as e:
        def clean_excel_string(s):
            if isinstance(s, str):
                return ILLEGAL_CHARACTERS_RE.sub("", s)
            return s
        for col in combined.select_dtypes(include=["object", "string"]):
            combined[col] = combined[col].apply(clean_excel_string)


def persona_already_done(output_excel: str, persona_id: str, model: str, expected_topics: Optional[int] = None) -> bool:
    """
    Returns True if the persona already has results in Excel.
    If expected_topics is provided, requires that many rows exist for that persona+model.
    """
    if not os.path.exists(output_excel):
        return False
    df = pd.read_excel(output_excel)
    if "persona_id" not in df.columns or "model" not in df.columns:
        return False
    sub = df[(df["persona_id"].astype(str) == str(persona_id)) & (df["model"].astype(str) == str(model))]
    if expected_topics is None:
        return len(sub) > 0
    return len(sub) >= expected_topics


def log_progress(log_path: str, payload: dict):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    payload = dict(payload)
    payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# --------------------------------------------------
# Output-path guard
# --------------------------------------------------
# A completed run's config has its `output_excel` key renamed to
# `disabled_output_excel`, so re-running it cannot overwrite the results that
# were published from it. Accept the historical `disabeled_` spelling too --
# both are present in the shipped configs.
DISABLED_OUTPUT_KEYS = ("disabled_output_excel", "disabeled_output_excel")


def resolve_output_excel(paths, config_path):
    if "output_excel" in paths:
        return paths["output_excel"]

    for key in DISABLED_OUTPUT_KEYS:
        if key in paths:
            raise SystemExit(
                f"\nOutput is disabled for {config_path}.\n"
                f"  This config declares '{key}' instead of 'output_excel', which\n"
                f"  deliberately prevents it from overwriting the completed run at:\n"
                f"      {paths[key]}\n"
                f"  To re-run it, rename that key to 'output_excel' in the config --\n"
                f"  and move or back up the existing workbook first, because\n"
                f"  --resume appends to it rather than replacing it.\n"
                f"  Ablations are unaffected: --ablation writes to its own path."
            )

    raise SystemExit(
        f"\n{config_path} declares no output path.\n"
        f"  Expected 'output_excel' under 'paths'; found: {sorted(paths)}"
    )


# 3 personas x 3 topics = 9 rows, which is enough to see whether the format
# holds and whether the token budget does. --limit overrides it.
DEFAULT_PILOT_PERSONAS = 3


def resolve_pilot_excel(paths, model):
    """Where a pilot run writes: a `pilot/` folder beside the real output.

    Deliberately bypasses :func:`resolve_output_excel`. A pilot cannot damage a
    completed run -- it writes somewhere else entirely -- so it must stay
    available for configs whose output has been disabled, which is exactly the
    state every published config ends up in.
    """
    declared = paths.get("output_excel")
    if declared is None:
        for key in DISABLED_OUTPUT_KEYS:
            if key in paths:
                declared = paths[key]
                break
    results_dir = os.path.dirname(declared) if declared else "results"
    return os.path.join(results_dir, "pilot", f"{model}_pilot.xlsx")


# --------------------------------------------------
# Main: loop over ALL personas, checkpoint after each persona
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="gpt-5.2.json")
    parser.add_argument("--resume", action="store_true", help="Skip personas that already exist in the Excel.")
    parser.add_argument("--print_prompt", action="store_true")
    parser.add_argument("--single_persona", type=str, default=None, help="Run only one persona folder.")
    parser.add_argument("--single_topic", type=str, default=None, help="Run only one topic (UBI|penalty|weight_loss).")
    parser.add_argument("--ablation", type=str, default=None, help="Specify changes to persona or question formulation")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N personas (deterministic: the persona list is sorted).")
    parser.add_argument("--pilot", action="store_true",
                        help=f"Smoke test: first {DEFAULT_PILOT_PERSONAS} personas (unless --limit), "
                             f"written to a pilot/ folder beside the real output.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    # Accepts both "model_size/X.json" and "configs/model_size/X.json"; the
    # SLURM scripts pass the latter because they glob for their config list.
    config_path = resolve_config_path(args.config)
    if not Path(config_path).exists():
        raise SystemExit(
            f"\nConfig not found: {args.config}\n"
            f"  Looked for: {config_path}\n"
            f"  Give a name under configs/ (model_size/Qwen3.5-9B.json) or a path."
        )
    print(f"Load config from {config_path}")
    config_list = load_json(config_path)
    config = config_list[0]

    model = config.get("model", "gpt-5.2")
    max_tokens_cfg = config.get("max_tokens") or config.get("max_completion_tokens")
    max_tries_cfg = config.get("max_tries", 1)
    port_cfg = config.get("port", 8000)
    temperature = config.get("temperature", 0.0)
    # Forwarded to vLLM's chat template -- this is how thinking is turned on for
    # Gemma 4 and kept on for Qwen3.5, which default in opposite directions.
    chat_template_kwargs = config.get("chat_template_kwargs")

    paths = config["paths"]
    personas_root = paths["personas_root"]
    template_file = paths["prompt_template_file"]
    if args.pilot:
        output_excel = resolve_pilot_excel(paths, model)
    elif args.ablation is None:
        output_excel = resolve_output_excel(paths, config_path)
    else:
        output_excel = f"results/ablations/{model}_{args.ablation}.xlsx"
    print("Saving results to path:", output_excel)
    if chat_template_kwargs:
        print("chat_template_kwargs:", safe_json_dumps(chat_template_kwargs))

    template_text = load_text(template_file)

    # client
    if model.startswith("gpt-"):
        client = OpenAI()
    elif model.startswith("gemini-"):
        if genai is None:
            raise SystemExit(
                f"{model} needs the google-genai package, which is not installed "
                f"in this environment. `pip install google-genai`."
            )
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    elif model.startswith("claude-"):
        if anthropic is None:
            raise SystemExit(
                f"{model} needs the anthropic package, which is not installed "
                f"in this environment. `pip install anthropic`."
            )
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    elif model.startswith("academic-ai"):
        client = AcademicAIClient(model=model)
    else:
        client = OpenAI(api_key="dummy", base_url=f"http://localhost:{port_cfg}/v1")

    # decide persona list
    if args.single_persona:
        persona_names = [args.single_persona]
    else:
        persona_names = [
            d for d in sorted(os.listdir(personas_root))
            if os.path.isdir(os.path.join(personas_root, d))
        ]

    if not persona_names:
        print(f"No persona folders found in: {personas_root}")
        return

    # Truncate before the resume filter, not after, so that `--limit N` always
    # means the same N personas however often the run is restarted.
    limit = args.limit if args.limit is not None else (DEFAULT_PILOT_PERSONAS if args.pilot else None)
    if limit is not None:
        persona_names = persona_names[:limit]
        print(f"Limited to the first {len(persona_names)} personas: {', '.join(persona_names)}")

    progress_log_path = os.path.join("results", "progress_log.jsonl")

    #persona_names = persona_names[:3]

    # run
    for i, pname in enumerate(persona_names, start=1):
        persona_dir = os.path.join(personas_root, pname)
        persona_id = pname  # folder name is persona_id

        # resume logic (assumes 3 topics in study; adjust if you change design)
        if args.resume and persona_already_done(output_excel, persona_id, model, expected_topics=None):
            print(f"[SKIP resume] persona {persona_id} already has results in Excel.")
            log_progress(progress_log_path, {"event": "skip", "persona_id": persona_id, "model": model})
            continue

        print(f"\n=== ({i}/{len(persona_names)}) Running persona: {persona_id} ===", flush=True)

        #try:
        persona_result = process_persona(
            client=client,
            persona_dir=persona_dir,
            template_text=template_text,
            model=model,
            max_tokens_cfg=max_tokens_cfg,
            max_tries_cfg=max_tries_cfg,
            temperature=temperature,
            single_topic=args.single_topic,
            print_prompt=args.print_prompt,
            ablation=args.ablation,
            results_path=output_excel,
            chat_template_kwargs=chat_template_kwargs,
        )

        if not persona_result or not persona_result.get("results_by_topic"):
            print(f"[WARN] No results produced for persona {persona_id}.")
            log_progress(progress_log_path, {"event": "no_results", "persona_id": persona_id, "model": model})
            continue
        
        def rec_print(s, depth):
            if isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, dict):
                        print(f"{' ' * depth}[DICTIONARY KEY] {k}:")
                        rec_print(v, depth + 4)
                    elif isinstance(v, list):
                        print(f"{' ' * depth}[DICTIONARY KEY FOR LIST] {k}:")
                        for i, item in enumerate(v):
                            print(f"{' ' * depth}Item {i}:")
                            rec_print(item, depth + 4)
                    else:
                        print(f"{' ' * depth}[SIMPLE DICTIONARY VALUE] {k}: {v}")

        #rec_print(persona_result, 0)

        #exit()

        # flatten rows
        new_rows: List[Dict[str, Any]] = []
        for topic, res in persona_result["results_by_topic"].items():
            new_rows.append(res)

        # checkpoint: append + dedupe + save Excel after this persona
        append_rows_to_excel(
            output_excel=output_excel,
            new_rows=new_rows,
            dedupe_keys=["persona_id", "topic", "model"],
        )

        print(f"[OK] Saved checkpoint Excel after persona {persona_id}: {output_excel}")
        log_progress(progress_log_path, {"event": "saved", "persona_id": persona_id, "model": model, "rows": len(new_rows)})
        '''
        except Exception as e:
            if isinstance(e, AssertionError):
                raise
            # Ensure we don't lose previous personas: Excel is already checkpointed.
            print(f"[ERROR] Persona {persona_id} failed with error: {e}")
            log_progress(progress_log_path, {"event": "error", "persona_id": persona_id, "model": model, "error": str(e)})
            # continue to next persona (optional). If you prefer to stop immediately, replace with: raise
            continue
        '''

    print("\nDone. Final Excel at:", output_excel)
    print("Progress log:", progress_log_path)
    if args.pilot:
        print(f"\nPilot finished. Check the token budget and format compliance with:\n"
              f"    python -m scripts.pipeline.pilot_report {output_excel}")


if __name__ == "__main__":
    main()
