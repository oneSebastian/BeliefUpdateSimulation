"""Loading comment rankings from the source files, in one convention.

Both sources index their rank columns by *shown slot*, but record different
things in them:

    human   rank_message_N_shown  = rank given to the comment in slot N
    model   rank_N                = id of the slot placed N-th

The model rows are converted on load (see :mod:`belief_update_sim.ranking`), so
everything this module returns uses the human convention: rank 1 = most
persuasive, indexed by shown slot.

Slot -> source comment is recovered per row: the human CSV carries
``message_N_shown`` and the results workbooks carry ``message_order_perm``, both
listing the source message id displayed at each position.
"""

import ast

import pandas as pd

from .config import MERGED_HUMAN_CSV, RESULTS_DIR
from .ranking import ranks_from_model_ordering

SLOTS = (1, 2, 3)
HUMAN = "Human"

# display name -> results file, in the order used throughout the paper
MODELS = {
    "GPT-5.2": "gpt5.2_results.xlsx",
    "GPT-5-Mini": "academic-ai-gpt-5-mini_all_personas_results.xlsx",
    "Gemini-3-Flash-Preview": "gemini-3-flash-preview_all_personas_results.xlsx",
    "Claude-Opus-4.6": "claude-opus-4-6_all_personas_results.xlsx",
    "Qwen-3-32B": "Qwen3_32B_all_personas_results.xlsx",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct_all_personas_results.xlsx",
}


def _int_or_none(value):
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def _as_slot_ranks(values):
    """{slot: rank}, or None if the triple is not a permutation of 1..3."""
    if sorted(v for v in values if v is not None) != list(SLOTS):
        return None
    return dict(zip(SLOTS, values))


def _perm(value):
    """message_order_perm: the source comment shown at each slot."""
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    try:
        return tuple(int(v) for v in ast.literal_eval(str(value)))
    except (ValueError, SyntaxError, TypeError):
        return None


def load_human_ranks():
    """(persona_id, topic) -> record. Already per-slot ranks; nothing converted."""
    df = pd.read_csv(MERGED_HUMAN_CSV)
    out = {}
    for _, row in df.iterrows():
        values = [_int_or_none(row.get(f"rank_message_{s}_shown")) for s in SLOTS]
        out[(str(row["persona_id"]), row["topic"])] = {
            "ranks": _as_slot_ranks(values),
            "shown": tuple(_int_or_none(row.get(f"message_{s}_shown")) for s in SLOTS),
            "package": row.get("package"),
        }
    return out


def load_model_ranks(path):
    """(persona_id, topic) -> record, converted into the human convention."""
    df = pd.read_excel(path)
    out = {}
    for _, row in df.iterrows():
        converted = ranks_from_model_ordering(row)
        out[(str(row["persona_id"]), row["topic"])] = {
            "ranks": _as_slot_ranks([converted[f"rank_{s}"] for s in SLOTS]),
            "shown": _perm(row.get("message_order_perm")),
            "package": row.get("package"),
        }
    return out


def load_all_sources():
    """{source name: records}, human first, then the models in paper order."""
    sources = {HUMAN: load_human_ranks()}
    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        sources[name] = load_model_ranks(path)
    return sources


def mean_rank_by_comment(records):
    """(topic, package, source_message_id) -> mean rank given to that comment.

    Ranks are recorded per shown slot, so each is attributed to whichever source
    comment occupied that slot for that participant.
    """
    totals = {}
    for (_, topic), rec in records.items():
        if rec["ranks"] is None or rec["shown"] is None:
            continue
        for slot, rank in rec["ranks"].items():
            source_id = rec["shown"][slot - 1]
            if source_id is None:
                continue
            totals.setdefault((topic, rec["package"], source_id), []).append(rank)
    return {key: sum(v) / len(v) for key, v in totals.items()}
