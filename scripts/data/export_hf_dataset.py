"""Export the belief-update data as a HuggingFace-ready JSONL dataset.

Three splits:
  main              human responses + the six headline models
  simulated_initial the same six models, but seeded with a model-generated
                    initial belief instead of the participant's real one
  ablations         post-training (OLMo), temperature, and prompt-component runs

Every belief field is emitted twice: raw_* exactly as recorded against the
statement as worded, and normalized_* flipped into the pro-topic frame (see
belief_update_sim/normalization.py). statement_polarity is carried per row so
the mapping between the two is always inspectable.

Usage:
    python -m scripts.data.export_hf_dataset
"""

import argparse
import json
import os

import pandas as pd

from belief_update_sim.config import (
    HF_DATASET_DIR, MERGED_HUMAN_CSV, PROJECT_ROOT, RESULTS_DIR,
)
from belief_update_sim.normalization import statement_polarity, normalize_value
from belief_update_sim.ranking import ranks_from_model_ordering
from belief_update_sim.response_parsing import repair_dataframe

ROOT = str(PROJECT_ROOT)
RESULTS = str(RESULTS_DIR)
ABLATIONS = os.path.join(RESULTS, "ablations")
HUMAN_CSV = str(MERGED_HUMAN_CSV)

# --- canonical model names -------------------------------------------------
# The `model` column, the filename token and the paper's name for a model all
# disagree (e.g. "academic-ai-gpt-5-mini" / "gpt-5-mini" / "GPT-5-Mini"), so the
# canonical name is assigned here rather than parsed from anything.
GPT52 = "GPT-5.2"
GPT5MINI = "GPT-5-Mini"
GEMINI = "Gemini-3-Flash-Preview"
CLAUDE = "Claude-Opus-4.6"
QWEN = "Qwen-3-32B"
LLAMA = "Llama-3.3-70B-Instruct"


def _r(name):
    return os.path.join(RESULTS, name)


def _a(name):
    return os.path.join(ABLATIONS, name)


# --- file registry ---------------------------------------------------------
# Explicit on purpose. Filename parsing silently miscategorises here: the token
# "Qwen3_32B_all_personas_results.xlsx" is a suffix of
# "template1_Qwen3_32B_all_personas_results.xlsx".
REGISTRY = [
    # main -----------------------------------------------------------------
    dict(path=_r("gpt5.2_results.xlsx"), split="main", model=GPT52),
    dict(path=_r("academic-ai-gpt-5-mini_all_personas_results.xlsx"), split="main", model=GPT5MINI),
    dict(path=_r("gemini-3-flash-preview_all_personas_results.xlsx"), split="main", model=GEMINI),
    dict(path=_r("claude-opus-4-6_all_personas_results.xlsx"), split="main", model=CLAUDE),
    dict(path=_r("Qwen3_32B_all_personas_results.xlsx"), split="main", model=QWEN),
    dict(path=_r("Llama-3.3-70B-Instruct_all_personas_results.xlsx"), split="main", model=LLAMA),

    # simulated_initial ----------------------------------------------------
    dict(path=_a("gpt-5.2_own-initial-belief.xlsx"), split="simulated_initial", model=GPT52),
    dict(path=_a("academic-ai-gpt-5-mini_own-initial-belief.xlsx"), split="simulated_initial", model=GPT5MINI),
    dict(path=_a("gemini-3-flash-preview_own-initial-belief.xlsx"), split="simulated_initial", model=GEMINI),
    dict(path=_a("claude-opus-4-6_own-initial-belief.xlsx"), split="simulated_initial", model=CLAUDE),
    dict(path=_a("Qwen3-32B_own-initial-belief.xlsx"), split="simulated_initial", model=QWEN),
    dict(path=_a("Llama-3.3-70B-Instruct_own-initial-belief.xlsx"), split="simulated_initial", model=LLAMA),

    # ablations: post-training ---------------------------------------------
    dict(path=_r("Olmo-3-32B-Think_all_personas_results.xlsx"), split="ablations",
         model="OLMo-3-32B-Think", ablation_type="post_training", condition="Think"),
    dict(path=_r("Olmo-3-32B-Think-SFT_all_personas_results.xlsx"), split="ablations",
         model="OLMo-3-32B-Think-SFT", ablation_type="post_training", condition="Think-SFT"),
    dict(path=_r("Olmo-3-32B-Think-DPO_all_personas_results.xlsx"), split="ablations",
         model="OLMo-3-32B-Think-DPO", ablation_type="post_training", condition="Think-DPO"),
    dict(path=_r("Olmo-3.1-32B-Instruct_all_personas_results.xlsx"), split="ablations",
         model="OLMo-3.1-32B-Instruct", ablation_type="post_training", condition="Instruct"),
    dict(path=_r("Olmo-3.1-32B-Instruct-SFT_all_personas_results.xlsx"), split="ablations",
         model="OLMo-3.1-32B-Instruct-SFT", ablation_type="post_training", condition="Instruct-SFT"),
    dict(path=_r("Olmo-3.1-32B-Instruct-DPO_all_personas_results.xlsx"), split="ablations",
         model="OLMo-3.1-32B-Instruct-DPO", ablation_type="post_training", condition="Instruct-DPO"),

    # ablations: temperature -----------------------------------------------
    dict(path=_a("academic-ai-gpt-5-mini_temperature_0.0.xlsx"), split="ablations",
         model=GPT5MINI, ablation_type="temperature", condition="t0.0", temperature=0.0),
    dict(path=_a("academic-ai-gpt-5-mini_temperature_2.0.xlsx"), split="ablations",
         model=GPT5MINI, ablation_type="temperature", condition="t2.0", temperature=2.0),
    dict(path=_a("Llama-3.3-70B-Instruct_t0.0.xlsx"), split="ablations",
         model=LLAMA, ablation_type="temperature", condition="t0.0", temperature=0.0),
    dict(path=_a("Llama-3.3-70B-Instruct_t2.0.xlsx"), split="ablations",
         model=LLAMA, ablation_type="temperature", condition="t2.0", temperature=2.0),
    dict(path=_a("Qwen3_32B_t0.0.xlsx"), split="ablations",
         model=QWEN, ablation_type="temperature", condition="t0.0", temperature=0.0),
    dict(path=_a("Qwen3_32B_t2.0.xlsx"), split="ablations",
         model=QWEN, ablation_type="temperature", condition="t2.0", temperature=2.0),

    # ablations: prompt component ------------------------------------------
    dict(path=_a("academic-ai-gpt-5-mini_no-demographic.xlsx"), split="ablations",
         model=GPT5MINI, ablation_type="prompt_component", condition="no-demographic"),
    dict(path=_a("academic-ai-gpt-5-mini_no-persona.xlsx"), split="ablations",
         model=GPT5MINI, ablation_type="prompt_component", condition="no-persona"),
    dict(path=_a("academic-ai-gpt-5-mini_no-personality.xlsx"), split="ablations",
         model=GPT5MINI, ablation_type="prompt_component", condition="no-personality"),
    dict(path=_a("Llama-3.3-70B-Instruct_no-demographic.xlsx"), split="ablations",
         model=LLAMA, ablation_type="prompt_component", condition="no-demographic"),
    dict(path=_a("Llama-3.3-70B-Instruct_no-persona.xlsx"), split="ablations",
         model=LLAMA, ablation_type="prompt_component", condition="no-persona"),
    dict(path=_a("Llama-3.3-70B-Instruct_no-personality.xlsx"), split="ablations",
         model=LLAMA, ablation_type="prompt_component", condition="no-personality"),
    dict(path=_a("Qwen3-32B_no-demographic.xlsx"), split="ablations",
         model=QWEN, ablation_type="prompt_component", condition="no-demographic"),
    dict(path=_a("Qwen3-32B_no-persona.xlsx"), split="ablations",
         model=QWEN, ablation_type="prompt_component", condition="no-persona"),
    dict(path=_a("Qwen3-32B_no-personality.xlsx"), split="ablations",
         model=QWEN, ablation_type="prompt_component", condition="no-personality"),
]

# Deliberately not published. Kept explicit so coverage checks stay exhaustive.
EXCLUDED = {
    # initial beliefs here are also present in the *_own-initial-belief runs
    "results/ablations/academic-ai-gpt-5-mini_probe-initial-belief.xlsx": "superseded by own-initial-belief",
    "results/ablations/claude-opus-4-6_probe-initial-belief.xlsx": "superseded by own-initial-belief",
    "results/ablations/gemini-3-flash-preview_probe-initial-belief.xlsx": "superseded by own-initial-belief",
    "results/ablations/gpt-5.2_probe-initial-belief.xlsx": "superseded by own-initial-belief",
    "results/ablations/Llama-3.3-70B-Instruct_probe-initial-belief.xlsx": "superseded by own-initial-belief",
    "results/ablations/Qwen3-32B_probe-initial-belief.xlsx": "superseded by own-initial-belief",
    "results/ablations/Llama-3.3-70B-Instruct_all-negative.xlsx": "statement-framing ablation, out of scope",
    "results/ablations/Llama-3.3-70B-Instruct_all-positive.xlsx": "statement-framing ablation, out of scope",
    "results/ablations/Qwen3-32B_all-negative.xlsx": "statement-framing ablation, out of scope",
    "results/ablations/Qwen3-32B_all-positive.xlsx": "statement-framing ablation, out of scope",
    "results/template1_Qwen3_32B_all_personas_results.xlsx": "prompt-template variant, out of scope",
    "results/all_personas_results.xlsx": "smoke test: 30 rows, 5 personas, two models in one file",
}

HUMAN_MODEL = "Human"

BIG5_COLS = [
    "BIG5_extraversion_score", "BIG5_agreeableness_score",
    "BIG5_conscientiousness_score", "BIG5_neuroticism_score", "BIG5_openness_score",
]

# The `demographic` blob is stored three different ways across the source files
# (9 decoded keys, 19 raw+decoded keys, and the human CSV's numeric codes).
# HF needs one struct schema, so every row is given the same canonical key set,
# built once per participant from the richest source and joined on persona_id.
#
# It describes the *participant*. Whether it was shown to the model depends on
# the run: see `condition`.
DEMO_KEYS = [
    "age", "gender", "ethnicity", "country_of_origin", "country_of_residency",
    "student_status", "employment_status", "occupation_field", "highest_qualification",
]
BIG5_TRAITS = ["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]

# 19-key decoded personas; the other files carry codes or a reduced key set.
PERSONA_SOURCE = _r("academic-ai-gpt-5-mini_all_personas_results.xlsx")

EMPTY_DEMOGRAPHIC = dict(
    {k: None for k in DEMO_KEYS},
    big5={t: None for t in BIG5_TRAITS},
    big5_items=[None] * 10,
)


def build_indices():
    """Canonical per-participant demographics, and message text by slot.

    Returns (demographic_by_persona, text_by_(topic, package, source_message_id)).
    The message index lets human rows carry the same message text the models
    saw; the human CSV records only which message was shown, not its wording.
    """
    df = pd.read_excel(PERSONA_SOURCE)
    demographic, texts = {}, {}
    for _, row in df.iterrows():
        pid = row["persona_id"]
        if pid not in demographic:
            blob = _maybe_json(row["demographic"]) or {}
            demographic[pid] = dict(
                {k: _jsonable(blob.get(k)) for k in DEMO_KEYS},
                big5={t: None for t in BIG5_TRAITS},
                big5_items=[None] * 10,
            )
        for msg in (_maybe_json(row["shown_messages"]) or []):
            if not isinstance(msg, dict):
                continue
            key = (row["topic"], msg.get("package"), msg.get("source_message_id"))
            texts.setdefault(key, msg.get("text"))

    human = pd.read_csv(HUMAN_CSV)
    for _, row in human.iterrows():
        entry = demographic.get(row["persona_id"])
        if entry is None:
            continue
        entry["big5"] = {t: _num(row.get(f"BIG5_{t}_score")) for t in BIG5_TRAITS}
        entry["big5_items"] = [_num(row.get(f"BIG_item_{i}")) for i in range(1, 11)]
    return demographic, texts


def canonical_messages(raw_messages, topic, package, texts):
    """One struct shape for shown_messages regardless of source.

    Model files store a list of dicts; the human CSV stores three bare message
    ids. Emitting both as-is gives list<struct> vs list<int64> in one split,
    which HF cannot cast.
    """
    out = []
    for slot, msg in enumerate(raw_messages or [], start=1):
        if isinstance(msg, dict):
            source_id = _num(msg.get("source_message_id"))
            out.append({
                "shown_comment_id": _num(msg.get("shown_comment_id")) or slot,
                "source_message_id": source_id,
                "package": _jsonable(msg.get("package")) or package,
                "text": _jsonable(msg.get("text")),
            })
        else:
            source_id = _num(msg)
            out.append({
                "shown_comment_id": slot,
                "source_message_id": source_id,
                "package": package,
                "text": texts.get((topic, package, source_id)),
            })
    return out


# Explicit schema. HF infers types from the first chunk of a JSONL file, which
# breaks here: `temperature` is null until row 7039 of the ablations split and
# `demographic.country_of_origin` is null throughout, so inference types them
# `null` and then fails to cast the real values. Declaring features in the card
# removes the guesswork.
_DEMO_FIELDS = [("age", "int64")] + [(k, "string") for k in DEMO_KEYS if k != "age"]

FEATURE_SPEC = [
    ("split", "string"),
    ("source", "string"),
    ("model", "string"),
    ("ablation_type", "string"),
    ("condition", "string"),
    ("temperature", "float64"),
    ("persona_id", "string"),
    ("topic", "string"),
    ("statement_formulation", "string"),
    ("statement_polarity", "int64"),
    ("package", "string"),
    ("message_order", "int64"),
    ("message_order_perm", ("sequence", "int64")),
    ("shown_messages", ("list", [
        ("shown_comment_id", "int64"),
        ("source_message_id", "int64"),
        ("package", "string"),
        ("text", "string"),
    ])),
    ("familiarity", "int64"),
    ("demographic", ("struct", _DEMO_FIELDS + [
        # two-item means, so half-points occur
        ("big5", ("struct", [(t, "float64") for t in BIG5_TRAITS])),
        ("big5_items", ("sequence", "int64")),
    ])),
    ("init_belief_source", "string"),
    ("raw_init_belief", "int64"),
    ("normalized_init_belief", "int64"),
    ("raw_new_belief", "int64"),
    ("normalized_new_belief", "int64"),
    ("raw_general_public_stance", "int64"),
    ("normalized_general_public_stance", "int64"),
    ("rank_1", "int64"),
    ("rank_2", "int64"),
    ("rank_3", "int64"),
    ("reasoning", "string"),
    ("raw_response", "string"),
]


def hf_features():
    """The same schema as a datasets.Features, for loading without inference."""
    from datasets import Features, Sequence, Value

    def build(spec):
        if isinstance(spec, str):
            return Value(spec)
        kind, inner = spec
        if kind == "sequence":
            return Sequence(Value(inner))
        if kind == "struct":
            return {name: build(t) for name, t in inner}
        if kind == "list":
            return [{name: build(t) for name, t in inner}]
        raise ValueError(kind)

    return Features({name: build(spec) for name, spec in FEATURE_SPEC})


def check_registry_covers_disk():
    """Fail loudly if a results file is neither exported nor explicitly excluded."""
    on_disk = set()
    for folder in (RESULTS, ABLATIONS):
        for name in os.listdir(folder):
            if name.endswith(".xlsx") and not name.startswith("~$"):
                on_disk.add(os.path.relpath(os.path.join(folder, name), ROOT).replace("\\", "/"))
    registered = {os.path.relpath(e["path"], ROOT).replace("\\", "/") for e in REGISTRY}
    excluded = set(EXCLUDED)

    unaccounted = on_disk - registered - excluded
    if unaccounted:
        raise RuntimeError(
            "results files are neither registered nor excluded:\n  "
            + "\n  ".join(sorted(unaccounted))
        )
    ghosts = (registered | excluded) - on_disk
    if ghosts:
        raise RuntimeError("registry references missing files:\n  " + "\n  ".join(sorted(ghosts)))
    return len(on_disk)


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _maybe_json(value):
    """Embedded JSON is stored as a string in the xlsx; parse to a real object."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, ValueError):
        return _jsonable(value)


def _num(value):
    value = _jsonable(value)
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else f


# gpt5.2_results.xlsx stores the model's reasoning under its own column name.
# Verified identical to the "reasoning" field parsed from raw_response on all
# 1173 rows, so this is a rename, not a merge.
COLUMN_ALIASES = {"free_text_gpt": "llm_reasoning"}


def load_llm_file(entry, indices=None):
    """Read one results file into export records, repairing null parses."""
    demographic_index, texts = indices if indices else ({}, {})
    df = pd.read_excel(entry["path"])
    for src, dst in COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    for col in ("llm_new_belief", "llm_general_public_stance", "llm_reasoning",
                "rank_1", "rank_2", "rank_3"):
        if col not in df.columns:
            df[col] = None

    repaired = repair_dataframe(df)

    records = []
    for _, row in df.iterrows():
        formulation = row["statement_formulation"]
        polarity = statement_polarity(formulation)
        init = _num(row.get("init_belief"))
        new = _num(row.get("llm_new_belief"))
        gps = _num(row.get("llm_general_public_stance"))
        persona_id = _jsonable(row["persona_id"])
        package = _jsonable(row.get("package"))
        records.append({
            "split": entry["split"],
            "source": "llm",
            "model": entry["model"],
            "ablation_type": entry.get("ablation_type"),
            "condition": entry.get("condition"),
            "temperature": entry.get("temperature"),
            "persona_id": persona_id,
            "topic": _jsonable(row["topic"]),
            "statement_formulation": formulation,
            "statement_polarity": polarity,
            "package": package,
            "message_order": _num(row.get("message_order_index")),
            "message_order_perm": _maybe_json(row.get("message_order_perm")),
            "shown_messages": canonical_messages(
                _maybe_json(row.get("shown_messages")), _jsonable(row["topic"]), package, texts),
            "familiarity": _num(row.get("familiarity")),
            "demographic": demographic_index.get(persona_id, EMPTY_DEMOGRAPHIC),
            "init_belief_source": ("simulated" if entry["split"] == "simulated_initial"
                                   else "participant"),
            "raw_init_belief": init,
            "normalized_init_belief": normalize_value(init, formulation),
            "raw_new_belief": new,
            "normalized_new_belief": normalize_value(new, formulation),
            "raw_general_public_stance": gps,
            "normalized_general_public_stance": normalize_value(gps, formulation),
            **ranks_from_model_ordering(row),
            "reasoning": _jsonable(row.get("llm_reasoning")),
            "raw_response": _jsonable(row.get("raw_response")),
        })
    return records, repaired


def load_human(indices=None):
    """Human participant responses, shaped like the model records."""
    demographic_index, texts = indices if indices else ({}, {})
    df = pd.read_csv(HUMAN_CSV)
    records = []
    for _, row in df.iterrows():
        formulation = row["statement_formulation"]
        polarity = statement_polarity(formulation)
        init = _num(row.get("initial_belief"))
        final = _num(row.get("final_belief"))
        persona_id = _jsonable(row["persona_id"])
        package = _jsonable(row.get("package"))
        records.append({
            "split": "main",
            "source": "human",
            "model": HUMAN_MODEL,
            "ablation_type": None,
            "condition": None,
            "temperature": None,
            "persona_id": persona_id,
            "topic": _jsonable(row["topic"]),
            "statement_formulation": formulation,
            "statement_polarity": polarity,
            "package": package,
            "message_order": _num(row.get("message_order_code")),
            "message_order_perm": None,
            "shown_messages": canonical_messages(
                [row.get(f"message_{i}_shown") for i in (1, 2, 3)],
                _jsonable(row["topic"]), package, texts),
            "familiarity": _num(row.get("familiarity")),
            "demographic": demographic_index.get(persona_id, EMPTY_DEMOGRAPHIC),
            "init_belief_source": "participant",
            "raw_init_belief": init,
            "normalized_init_belief": normalize_value(init, formulation),
            "raw_new_belief": final,
            "normalized_new_belief": normalize_value(final, formulation),
            "raw_general_public_stance": None,
            "normalized_general_public_stance": None,
            "rank_1": _num(row.get("rank_message_1_shown")),
            "rank_2": _num(row.get("rank_message_2_shown")),
            "rank_3": _num(row.get("rank_message_3_shown")),
            "reasoning": _jsonable(row.get("text_response")),
            "raw_response": None,
        })
    return records


def cross_check_human_normalization(records):
    """The merged CSV ships its own normalized columns; confirm we agree."""
    df = pd.read_csv(HUMAN_CSV)
    by_key = {(r["persona_id"], r["topic"]): r for r in records}
    mismatches = 0
    for _, row in df.iterrows():
        rec = by_key.get((row["persona_id"], row["topic"]))
        if rec is None:
            continue
        for ours, theirs in (("normalized_init_belief", "initial_belief_normalized"),
                             ("normalized_new_belief", "final_belief_normalized")):
            a, b = rec[ours], _num(row.get(theirs))
            if a is not None and b is not None and a != b:
                mismatches += 1
    return mismatches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HF_DATASET_DIR))
    args = ap.parse_args()

    n_files = check_registry_covers_disk()
    print(f"registry check ok: {n_files} results files, "
          f"{len(REGISTRY)} exported, {len(EXCLUDED)} excluded")

    os.makedirs(args.out, exist_ok=True)
    splits = {"main": [], "simulated_initial": [], "ablations": []}

    indices = build_indices()
    print(f"persona index: {len(indices[0])} participants, "
          f"{len(indices[1])} distinct messages")

    human = load_human(indices)
    splits["main"].extend(human)
    mismatches = cross_check_human_normalization(human)
    print(f"human rows: {len(human)}  normalization cross-check mismatches: {mismatches}")
    if mismatches:
        raise RuntimeError("normalized human beliefs disagree with the merged CSV")

    total_repaired = 0
    for entry in REGISTRY:
        records, repaired = load_llm_file(entry, indices)
        splits[entry["split"]].extend(records)
        total_repaired += repaired
        tag = entry.get("condition") or "-"
        print(f"  {entry['split']:<18} {entry['model']:<26} {tag:<16} "
              f"rows={len(records):<6} repaired={repaired}")

    print(f"\ntotal rows repaired from raw_response: {total_repaired}")

    for split, records in splits.items():
        path = os.path.join(args.out, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(path) / 1e6
        print(f"wrote {path}  rows={len(records)}  {size_mb:.1f}MB")

    card = os.path.join(args.out, "README.md")
    with open(card, "w", encoding="utf-8") as fh:
        fh.write(dataset_card(splits))
    print(f"wrote {card}")


def dataset_card(splits):
    counts = {k: len(v) for k, v in splits.items()}
    models = sorted({e["model"] for e in REGISTRY if e["split"] == "main"})
    return f"""---
license: cc-by-4.0
language:
  - en
annotations_creators:
  - crowdsourced
tags:
  - belief-update
  - persuasion
  - social-simulation
  - llm-agents
  - human-subjects
configs:
  - config_name: default
    data_files:
      - split: main
        path: main.jsonl
      - split: simulated_initial
        path: simulated_initial.jsonl
      - split: ablations
        path: ablations.jsonl
---

# Belief Updates: Humans and Simulated LLM Agents

Human and LLM responses to the same belief-update task. 391 participants each
gave an initial belief on three topics (UBI, penalty shootouts, weight-loss
drugs), read three comments, then gave an updated belief and ranked the
comments by persuasiveness. The same material was given to LLMs conditioned on
each participant's persona.

## Splits

| split | rows | contents |
|---|---|---|
| `main` | {counts['main']} | human responses + {len(models)} models, each seeded with the participant's real initial belief |
| `simulated_initial` | {counts['simulated_initial']} | the same models, seeded with a model-generated initial belief |
| `ablations` | {counts['ablations']} | post-training (OLMo-3), temperature, and prompt-component runs |

Splits are disjoint. `source` is `human` or `llm`; `model` is `Human` on human
rows.

## Belief sign convention

Each topic was worded either for or against it, so a belief recorded against a
negatively-worded statement sits in the opposite frame from one recorded
against a positively-worded statement. Every belief field is therefore given
twice:

- `raw_*` — exactly as recorded, against the statement as worded
- `normalized_*` — flipped into the pro-topic frame, so positive always means
  agreement with the topic (pro-UBI, pro-shootouts, pro-weight-loss-drugs)

`statement_polarity` (+1/-1) relates them: `normalized_x == raw_x *
statement_polarity`. **Use the normalized fields when pooling or comparing
across formulations.** Ranks are never flipped — they index which message was
shown, not a polarity.

## Ranking convention

`rank_N` is **the rank the respondent gave to the comment in slot N**, where
1 is most persuasive and 3 least. Slot N refers to `shown_messages[N-1]`, so
join through `shown_comment_id` to recover which source comment was ranked.

The two sources recorded this differently at collection time: the human survey
asked participants to assign a position to each message, while the models were
asked to list comment IDs from most to least convincing — inverse permutations
of one another. **The model rows are converted on export**, so a single
convention holds across every row and `rank_N` is directly comparable between
humans and models.

This matters more than it looks: for three items, four of the six possible
permutations are their own inverse, so the two conventions agree on two thirds
of rows. Mixing them yields results that look reasonable instead of failing
loudly.

## Fields

| field | notes |
|---|---|
| `persona_id` | pseudonym (`P0001`-`P0429`); joins human and model rows for the same participant |
| `topic` | `UBI`, `penalty`, `weight_loss` |
| `statement_formulation`, `statement_polarity` | wording shown, and its sign |
| `raw_init_belief` / `normalized_init_belief` | belief before reading the comments, on -2..2 |
| `raw_new_belief` / `normalized_new_belief` | belief after |
| `raw_general_public_stance` / `normalized_general_public_stance` | model's estimate of public opinion; null for humans |
| `init_belief_source` | `participant` or `simulated` |
| `rank_1/2/3` | `rank_N` is the rank the respondent gave the comment in slot N; 1 = most persuasive (see below) |
| `reasoning` | model's free-text justification, or the participant's text response |
| `shown_messages`, `message_order`, `package` | which comments were shown, in what order |
| `demographic` | the participant's attributes; whether they were shown to the model depends on `condition` |
| `raw_response` | verbatim model output, before parsing |
| `ablation_type`, `condition`, `temperature` | populated on the `ablations` split |

## Known limitations

- The OLMo-3 Think checkpoints' structured fields were originally null: those
  models emit a closing `</think>` with no opening tag, which defeated the
  generation-time parser. They are recovered here from `raw_response`; 3 rows
  across the three checkpoints remain unrecovered.
- One OLMo-3-32B-Think response was truncated at 32,767 characters by Excel's
  per-cell limit when it was written, and its trailing JSON is unrecoverable.
- Temperature-2.0 runs have genuine parse failures (Llama 2.0%, Qwen 0.9%),
  left null rather than imputed.
- Gemini-3-Flash-Preview has one null response in `main`.

## Participant privacy

`persona_id` is a pseudonym of the form `P0001`–`P0429`, assigned by a seeded
shuffle so the numbering carries no information about recruitment order or any
other participant attribute. The recruitment-platform identifiers it replaced
were removed from every file, and the key linking the two is held privately and
is not published. Platform-internal session tokens and per-participant
timestamps were dropped from the source exports for the same reason.

The same pseudonym is used consistently everywhere, so human and model rows for
one participant still join on `persona_id`.

Self-reported demographics are retained, since they are the experimental
conditioning variable. Free-text fields were screened for self-identifying
content. Note that this is pseudonymisation, not anonymisation: demographics
are quasi-identifiers, and the free text is participants' own writing.

## Not included

Probe-initial-belief runs (those beliefs also appear in `simulated_initial`),
statement-framing ablations, the persona-extremity and prompt-template
variants, and a 30-row smoke test. All remain in the source repository.

The verbatim prompt sent to each model is also omitted — it is largely the same
template repeated per row, and accounted for roughly a third of the total size.
It can be reconstructed from `prompt_templates/` in the source repository
together with `demographic`, `shown_messages` and `condition`.

## Ethics and consent

Prior to data collection this project has been reviewed by the research ethics
committee of the Interdisciplinary Transformation University Austria under the
case number 2025-09.

Participants were recruited through Prolific and gave informed consent before
any data was collected. They were told that the study investigates the behaviour
of humans and LLM agents, that they would not interact with an LLM at any point,
and that their demographic information would be used to condition LLM agents on
a comparable distribution in a later study — the study this dataset comes from.

They were informed that pseudonymised data would be released publicly and would
not permit their identification, and that they could discontinue the study at
any time without penalty. Participants who did not consent were routed out
before any responses were recorded, and returned submissions are excluded
(391 of 400 recruited participants remain).

**When using this dataset, do not attempt to re-identify participants**, either
directly or by linking these records against any other dataset. The demographic
fields are quasi-identifiers and the free text is participants' own writing.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The code that
produced it is MIT-licensed in the source repository.
"""


if __name__ == "__main__":
    main()
