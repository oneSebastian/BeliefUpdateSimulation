"""
Temporary script to backfill missing initial_belief values in probe-initial-belief xlsx files
by re-parsing the raw_response column with the fixed extraction logic.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd


XLSX_FILES = [
    "results/ablations/gpt-5.2_probe-initial-belief.xlsx",
    "results/ablations/gemini-3-flash-preview_probe-initial-belief.xlsx",
    "results/ablations/claude-opus-4-6_probe-initial-belief.xlsx",
]

VALID_BELIEFS = {-2, -1, 0, 1, 2}


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
    if cleaned.startswith("'''") and cleaned.endswith("'''"):
        cleaned = cleaned[3:-3].strip()
    elif cleaned.startswith('"""') and cleaned.endswith('"""'):
        cleaned = cleaned[3:-3].strip()
    elif (cleaned.startswith('"') and cleaned.endswith('"')) or \
         (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def parse_initial_belief(raw_response: str):
    if not isinstance(raw_response, str):
        return None
    cleaned = _clean_json_text(raw_response)
    try:
        data = json.loads(cleaned)
        value = int(data["belief"])
        return value if value in VALID_BELIEFS else None
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def process_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"SKIP (not found): {path}")
        return

    df = pd.read_excel(path)
    missing_mask = df["initial_belief"].isna()
    n_missing = missing_mask.sum()
    print(f"\n{path}: {n_missing} missing out of {len(df)}")

    if n_missing == 0:
        return

    recovered = 0
    still_missing = 0
    for idx in df[missing_mask].index:
        raw = df.at[idx, "raw_response"]
        belief = parse_initial_belief(raw)
        if belief is not None:
            df.at[idx, "initial_belief"] = belief
            recovered += 1
        else:
            still_missing += 1
            print(f"  [WARN] Could not parse row {idx} (persona={df.at[idx, 'persona_id']}, topic={df.at[idx, 'topic']})")
            print(f"         raw_response[:200]: {str(raw)[:200]}")

    df.to_excel(path, index=False)
    print(f"  Recovered: {recovered}, still missing: {still_missing} — saved.")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else XLSX_FILES
    for f in targets:
        process_file(f)
