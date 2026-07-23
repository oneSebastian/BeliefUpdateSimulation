"""Comment-ranking conventions, and the conversion between them.

The two data sources record rankings in opposite conventions, and both store a
permutation of {1, 2, 3} in three columns, so nothing about the values reveals
which convention is in play.

humans
    oTree's ranking page is a grid with one row per message and columns
    1st/2nd/3rd, so ``rank_topic_T_M`` (surfaced as ``rank_message_M_shown``)
    holds the rank *given to* the message in slot M.

        rank_N = R  ->  "the comment in slot N was placed R-th"

models
    The prompt asks for the shown comment IDs ordered most- to least-convincing,
    and run_agent stores ``rank_1 = ranking[0]``.

        rank_i = C  ->  "the comment placed i-th was the one in slot C"

These are inverse permutations: index and value swap roles. Four of the six
permutations of three items are their own inverse (123, 213, 132, 321), so the
two readings coincide on two thirds of rows and differ only on 231 and 312.
Mixing them therefore produces plausible-looking wrong answers rather than an
error -- a naive comparison can report perfect agreement for a pair that in fact
disagrees.

Anything that compares model rankings against human ones must first put the
model rows into the human convention with :func:`ranks_from_model_ordering`.
Code that only round-trips a model's own output (parsing, repair) should leave
it in the native ordering.
"""

SLOTS = (1, 2, 3)
NULL_RANKS = {f"rank_{slot}": None for slot in SLOTS}


def ranks_from_model_ordering(row):
    """Convert a model's ordering into per-slot ranks.

    ``row`` is any mapping with ``rank_1``/``rank_2``/``rank_3`` (a dict or a
    pandas Series). Returns a dict in the same three keys, holding the rank given
    to the comment in each slot, 1 = most persuasive.

    Rankings that are incomplete or not a permutation of {1, 2, 3} -- the parse
    failures in the temperature-2.0 runs, for instance -- yield all-null rather
    than a partial mapping, so a bad row cannot masquerade as a good one.
    """
    ordering = [_as_int(row.get(f"rank_{slot}")) for slot in SLOTS]
    if sorted(x for x in ordering if x is not None) != list(SLOTS):
        return dict(NULL_RANKS)

    by_slot = {}
    for position, shown_id in enumerate(ordering, start=1):
        by_slot[shown_id] = position
    # fixed key order, so field order does not vary row to row on export
    return {f"rank_{slot}": by_slot[slot] for slot in SLOTS}


def _as_int(value):
    if value is None:
        return None
    if isinstance(value, float) and value != value:      # NaN
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None
