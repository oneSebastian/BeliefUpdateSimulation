"""Recover structured fields from a model's raw_response.

Mirrors parse_json_output() in figures/ablations/post_training.py, which is the
parser that already copes with reasoning models. run_agent.py's own parser
strips reasoning with `<think>.*?</think>`, requiring both tags; the Olmo-3
Think checkpoints emit only the closing `</think>`, so nothing was stripped,
json.loads() saw the whole chain-of-thought and every parsed column was written
null. Stripping `.*?</think>` instead recovers those rows from raw_response.

Also coerces string-valued fields: the Think checkpoints emit
{"new_belief": "0", "ranking": ["2","1","3"]} where the Instruct models emit
integers.
"""

import json
import re
import unicodedata

_THINK_PREFIX = re.compile(r".*?</think>", re.DOTALL | re.IGNORECASE)


def parse_json_output(output):
    """Return a dict of parsed fields, or None if nothing usable is present.

    Keys: new_belief, ranking, reasoning, general_public_stance.
    """
    if output is None:
        return None
    text = str(output)
    if isinstance(output, float) and output != output:  # NaN
        return None

    cleaned = _THINK_PREFIX.sub("", text, count=1).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    try:
        new_belief = int(data["new_belief"])
        general_public_stance = int(data["general_public_stance"])
        ranking = [int(r) for r in data["ranking"]]
        reasoning = unicodedata.normalize("NFKC", str(data["reasoning"]))
    except (KeyError, TypeError, ValueError):
        return None

    if len(ranking) != 3:
        return None

    return {
        "new_belief": new_belief,
        "ranking": ranking,
        "reasoning": reasoning,
        "general_public_stance": general_public_stance,
    }


SCALAR_TARGETS = {
    "llm_new_belief": "new_belief",
    "llm_general_public_stance": "general_public_stance",
    "llm_reasoning": "reasoning",
}
RANK_TARGETS = ["rank_1", "rank_2", "rank_3"]


def repair_dataframe(df):
    """Fill null parsed columns from raw_response, in place. Returns row count.

    Only writes where the existing value is null, so runs that parsed correctly
    at generation time are never overwritten.

    Target columns are cast to object first: a fully-null column loads as
    float64, and writing reasoning text into it would otherwise hit pandas'
    incompatible-dtype coercion.
    """
    import pandas as pd

    if "raw_response" not in df.columns:
        return 0

    cols = [c for c in list(SCALAR_TARGETS) + RANK_TARGETS if c in df.columns]
    for col in cols:
        df[col] = df[col].astype(object)

    repaired = 0
    for idx in df.index:
        if not any(pd.isna(df.at[idx, c]) for c in cols):
            continue
        parsed = parse_json_output(df.at[idx, "raw_response"])
        if parsed is None:
            continue
        changed = False
        for col, key in SCALAR_TARGETS.items():
            if col in cols and pd.isna(df.at[idx, col]):
                df.at[idx, col] = parsed[key]
                changed = True
        for i, col in enumerate(RANK_TARGETS):
            if col in cols and pd.isna(df.at[idx, col]):
                df.at[idx, col] = parsed["ranking"][i]
                changed = True
        repaired += bool(changed)
    return repaired
