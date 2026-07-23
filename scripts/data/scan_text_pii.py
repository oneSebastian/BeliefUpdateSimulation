"""Scan free-text fields for self-identifying content.

Complements pseudonymize.py: replacing participant ids does nothing about a
participant who typed their name, employer, or email into a free-text box, or
about a model that echoed such content back.

Fields scanned:
  human  text_response        (merged CSV / hf_dataset reasoning on human rows)
  llm    llm_reasoning        (results xlsx / hf_dataset reasoning)
  llm    raw_response         (results xlsx -- verbatim model output)
  human  study_data.json      free-text answers per participant

Output is a triage report: counts per pattern, plus redacted samples with a
locator so each hit can be reviewed by hand. Nothing is modified.

    python -m scripts.data.scan_text_pii
    python -m scripts.data.scan_text_pii --samples 5 --include-raw-response
"""

import argparse
import json
import re
from collections import defaultdict

import pandas as pd

from belief_update_sim.config import DATA_DIR, HF_DATASET_DIR, MERGED_HUMAN_CSV, RESULTS_DIR

# Ordered roughly by how strongly each implies real identifying content.
PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")),
    ("url", re.compile(r"https?://\S+|www\.\S+")),
    ("social_handle", re.compile(r"(?<![\w@])@[A-Za-z][\w.]{2,}\b")),
    ("uk_postcode", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")),
    ("uk_nino", re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b")),
    ("long_digit_run", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
    ("date_of_birth", re.compile(r"\b(?:born|dob|date of birth)\b[^.\n]{0,30}\d{4}", re.I)),
    ("self_names_me", re.compile(r"\b(?:my name is|i am called|call me|i'm called)\b", re.I)),
    ("names_employer", re.compile(r"\b(?:i work (?:at|for)|my employer|my company|my boss)\b", re.I)),
    ("names_location", re.compile(r"\b(?:i live in|i'm from|i am from|my address|my postcode)\b", re.I)),
    ("prolific_mention", re.compile(r"\bprolific\b", re.I)),
]


def redact(text, match, width=55):
    """Show context around a hit with the hit itself masked."""
    s, e = match.span()
    left = text[max(0, s - width):s].replace("\n", " ")
    right = text[e:e + width].replace("\n", " ")
    body = match.group(0)
    masked = body[:2] + "*" * max(0, len(body) - 3) + body[-1:] if len(body) > 4 else "***"
    return f"...{left}[{masked}]{right}..."


class Report:
    def __init__(self, n_samples):
        self.counts = defaultdict(lambda: defaultdict(int))   # field -> pattern -> n
        self.samples = defaultdict(list)                      # (field,pattern) -> [(loc, snippet)]
        self.scanned = defaultdict(int)
        self.n_samples = n_samples

    def scan(self, field, locator, text):
        if not isinstance(text, str) or not text.strip():
            return
        self.scanned[field] += 1
        for name, pat in PATTERNS:
            for m in pat.finditer(text):
                self.counts[field][name] += 1
                key = (field, name)
                if len(self.samples[key]) < self.n_samples:
                    self.samples[key].append((locator, redact(text, m)))
                break  # one hit per pattern per text is enough for triage

    def render(self):
        print("\n" + "=" * 78)
        print("FREE-TEXT PII TRIAGE")
        print("=" * 78)
        for field in sorted(self.counts):
            total = sum(self.counts[field].values())
            print(f"\n{field}   ({self.scanned[field]} non-empty texts scanned, "
                  f"{total} flagged)")
            for name, n in sorted(self.counts[field].items(), key=lambda x: -x[1]):
                print(f"    {n:>6}  {name}")
        clean = [f for f in sorted(self.scanned) if not self.counts[f]]
        for f in clean:
            print(f"\n{f}   ({self.scanned[f]} texts scanned)  -- no matches")

        print("\n" + "-" * 78)
        print("SAMPLES (hit masked; review these by hand)")
        print("-" * 78)
        for (field, name) in sorted(self.samples):
            print(f"\n[{field} :: {name}]")
            for loc, snip in self.samples[(field, name)]:
                print(f"   {loc}\n      {snip}")


def scan_human_csv(rep):
    if not MERGED_HUMAN_CSV.exists():
        return
    df = pd.read_csv(MERGED_HUMAN_CSV)
    if "text_response" not in df.columns:
        return
    for _, row in df.iterrows():
        rep.scan("human.text_response",
                 f"{row['persona_id']} / {row.get('topic')}",
                 row["text_response"])


def scan_results(rep, include_raw):
    for path in sorted(RESULTS_DIR.rglob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        cols = ["persona_id", "topic", "llm_reasoning"] + (["raw_response"] if include_raw else [])
        try:
            df = pd.read_excel(path, usecols=lambda c: c in cols)
        except Exception:
            continue
        for _, row in df.iterrows():
            loc = f"{path.name} :: {row.get('persona_id')} / {row.get('topic')}"
            if "llm_reasoning" in df.columns:
                rep.scan("llm.reasoning", loc, row.get("llm_reasoning"))
            if include_raw and "raw_response" in df.columns:
                rep.scan("llm.raw_response", loc, row.get("raw_response"))


def scan_study_data(rep):
    d = DATA_DIR / "prolific_data"
    if not d.is_dir():
        return
    for pdir in sorted(d.iterdir()):
        f = pdir / "study_data.json"
        if not f.exists():
            continue
        blob = json.loads(f.read_text(encoding="utf-8"))

        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}" if path else k)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")
            elif isinstance(node, str) and len(node) > 15:
                rep.scan("human.study_data", f"{pdir.name} :: {path}", node)

        walk(blob)


def scan_hf(rep):
    f = HF_DATASET_DIR / "main.jsonl"
    if not f.exists():
        return
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("source") == "human":
                rep.scan("hf.human_reasoning",
                         f"{rec['persona_id']} / {rec['topic']}", rec.get("reasoning"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--include-raw-response", action="store_true",
                    help="also scan verbatim model output (slow, ~200MB of text)")
    args = ap.parse_args()

    rep = Report(args.samples)
    print("scanning human free text (merged CSV) ...")
    scan_human_csv(rep)
    print("scanning participant study_data.json ...")
    scan_study_data(rep)
    print("scanning hf_dataset human reasoning ...")
    scan_hf(rep)
    print("scanning model reasoning in results/*.xlsx ...")
    scan_results(rep, args.include_raw_response)
    rep.render()


if __name__ == "__main__":
    main()
