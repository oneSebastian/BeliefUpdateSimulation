"""
Load demographic data from data/prolific_data/<pid>/demographic.json files,
compile into a DataFrame, and print breakdowns by age bin, sex, and ethnicity.

    python -m scripts.figures.demographics.load_json
"""
import json
import os

import pandas as pd

from belief_update_sim.config import PROLIFIC_DATA_DIR
AGE_BINS   = [18, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def load_demographics():
    rows = []
    for pid in os.listdir(PROLIFIC_DATA_DIR):
        path = os.path.join(PROLIFIC_DATA_DIR, pid, "demographic.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "participant_id": d["participant_id"],
            "age":            d["age"],
            "sex":            d["gender"],
            "ethnicity":      d["ethnicity"],
        })
    df = pd.DataFrame(rows)
    df["age_bin"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


if __name__ == "__main__":
    df = load_demographics()
    n = len(df)
    print(f"N = {n}\n")

    print("--- Age ---")
    print(df.groupby("age_bin", observed=False).size().to_string())
    print()

    print("--- Sex ---")
    print(df.groupby("sex").size().to_string())
    print()

    print("--- Ethnicity ---")
    print(df.groupby("ethnicity").size().to_string())