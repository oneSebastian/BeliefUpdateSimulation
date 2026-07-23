"""Replace Prolific participant IDs with stable pseudonyms across the repository.

The IDs appear in four different shapes, and all four must move together or
joins break (e.g. inspect_rankings.py matches prolific_data/ directory names
against the persona_id column):

  1. directory names   data/prolific_data/<id>/
  2. JSON content      demographic.json / study_data.json  ("participant_id")
  3. text columns      CSV and JSONL  (participant.label / persona_id)
  4. xlsx cells        persona_id column + the embedded `demographic` blob

The mapping is the re-identification key. It is written OUTSIDE the repository
and the script refuses to write it anywhere under PROJECT_ROOT. Re-runs reload
the existing map, so assignments never change once issued.

    python -m scripts.data.pseudonymize --dry-run     # report, change nothing
    python -m scripts.data.pseudonymize               # apply, then verify
    python -m scripts.data.pseudonymize --verify-only # just check for leftovers
"""

import argparse
import csv
import os
import random
import re
import sys

from openpyxl import load_workbook

from belief_update_sim.config import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

DEFAULT_MAP = PROJECT_ROOT.parent / "BeliefUpdateSimulation-private" / "participant_id_map.csv"

PROLIFIC_DATA = DATA_DIR / "prolific_data"
HF_DATASET = DATA_DIR / "hf_dataset"
CLEANED_CSV = DATA_DIR / "cleaned_for_llm_391_participant_otree.csv"
MERGED_CSV = RESULTS_DIR / "merged_llm_participants_data_with_normalized_beliefs.csv"
OTREE_XLSX = [DATA_DIR / "400_participant_otree.xlsx",
              DATA_DIR / "cleaned_391_participant_otree.xlsx"]

# A participant id is a 24-char alphanumeric token. Excluded on purpose:
#   {{%PROLIFIC_PID%}}  an unfilled Prolific template placeholder (a preview row)
#   nan                 a missing label
ID_TOKEN = re.compile(r"^[0-9A-Za-z]{24}$")
NOT_AN_ID = {"nan", "", "{{%PROLIFIC_PID%}}"}

SEED = 20260722
PSEUDONYM_FMT = "P{:04d}"


# --------------------------------------------------------------------------
# collecting the original ids
# --------------------------------------------------------------------------

def _tokens_from_text(text):
    return {t for t in re.findall(r"\b[0-9A-Za-z]{24}\b", text)}


def _otree_label_column(path):
    """The raw oTree exports cram every field into one comma-joined cell."""
    import io

    import pandas as pd
    raw = pd.read_excel(path, sheet_name=0, header=None)
    text = "\n".join(",".join("" if pd.isna(v) else str(v) for v in row) for row in raw.values)
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    col = [c for c in df.columns if c.endswith("participant.label")]
    return set(df[col[0]].astype(str)) if col else set()


def collect_original_ids():
    """Union of every participant id, read from ID-BEARING COLUMNS only.

    Deliberately the union, not just the analysed 391: the 400-participant
    oTree export carries ~40 additional ids (excluded/returned participants)
    that would otherwise survive as real Prolific ids in a published file.

    Equally deliberately NOT a free-text token scan. The temperature-2.0 runs
    emit degenerate token salad, and 7 of those strings happen to be 24
    alphanumeric characters (e.g. "RepresentsPayloadHeaders"). Harvesting ids
    by pattern would map them as if they were participants and corrupt
    raw_response. Ids come from structured columns; only the *replacement*
    is a whole-text pass.
    """
    import pandas as pd
    ids = set()

    if PROLIFIC_DATA.is_dir():
        ids |= {d.name for d in PROLIFIC_DATA.iterdir()
                if d.is_dir() and ID_TOKEN.match(d.name)}

    if CLEANED_CSV.exists():
        ids |= set(pd.read_csv(CLEANED_CSV)["participant.label"].astype(str))
    if MERGED_CSV.exists():
        ids |= set(pd.read_csv(MERGED_CSV)["persona_id"].astype(str))

    for path in sorted(RESULTS_DIR.rglob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        try:
            ids |= set(pd.read_excel(path, usecols=["persona_id"])["persona_id"].astype(str))
        except Exception:
            continue

    for path in OTREE_XLSX:
        if path.exists():
            ids |= _otree_label_column(path)

    if HF_DATASET.is_dir():
        import json
        for f in HF_DATASET.glob("*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    ids.add(json.loads(line)["persona_id"])

    return {i for i in ids if i not in NOT_AN_ID and ID_TOKEN.match(str(i))}


# --------------------------------------------------------------------------
# the mapping
# --------------------------------------------------------------------------

def load_or_create_map(map_path, ids, dry_run):
    """Stable bijective old->new map. Existing assignments are never reshuffled."""
    resolved = map_path.resolve()
    if PROJECT_ROOT.resolve() in resolved.parents or resolved == PROJECT_ROOT.resolve():
        raise SystemExit(
            f"refusing to write the re-identification key inside the repository:\n"
            f"  {resolved}\n"
            f"This repo is headed for publication; keep the key outside it."
        )

    mapping = {}
    if resolved.exists():
        with open(resolved, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                mapping[row["original_id"]] = row["pseudonym"]
        print(f"loaded existing map: {len(mapping)} assignments from {resolved}")

    new_ids = sorted(ids - set(mapping))
    if new_ids:
        # shuffle before numbering so the pseudonym does not encode file order
        rng = random.Random(SEED)
        shuffled = new_ids[:]
        rng.shuffle(shuffled)
        used = set(mapping.values())
        n = 1
        for original in shuffled:
            while PSEUDONYM_FMT.format(n) in used:
                n += 1
            mapping[original] = PSEUDONYM_FMT.format(n)
            used.add(mapping[original])
        print(f"assigned {len(new_ids)} new pseudonyms")

    assert len(set(mapping.values())) == len(mapping), "pseudonym collision"

    if new_ids and not dry_run:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["original_id", "pseudonym"])
            for original in sorted(mapping):
                w.writerow([original, mapping[original]])
        print(f"wrote map -> {resolved}")
    elif new_ids:
        print(f"[dry-run] would write map -> {resolved}")

    return mapping


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------

def build_pattern(mapping):
    # no id is a substring of another (verified), so plain alternation is safe
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(mapping)) + r")\b")


def rewrite_text_file(path, pattern, mapping, dry_run):
    text = path.read_text(encoding="utf-8", errors="ignore")
    new, n = pattern.subn(lambda m: mapping[m.group(0)], text)
    if n and not dry_run:
        path.write_text(new, encoding="utf-8")
    return n


def rewrite_xlsx(path, pattern, mapping, dry_run):
    wb = load_workbook(path)
    n = 0
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and len(cell.value) >= 24:
                    new, k = pattern.subn(lambda m: mapping[m.group(0)], cell.value)
                    if k:
                        cell.value = new
                        n += k
    if n and not dry_run:
        wb.save(path)
    wb.close()
    return n


def text_targets():
    for p in (CLEANED_CSV, MERGED_CSV):
        if p.exists():
            yield p
    if HF_DATASET.is_dir():
        yield from sorted(HF_DATASET.glob("*.jsonl"))
    if PROLIFIC_DATA.is_dir():
        yield from sorted(PROLIFIC_DATA.glob("*/*.json"))


def xlsx_targets():
    for p in sorted(RESULTS_DIR.rglob("*.xlsx")):
        if not p.name.startswith("~$"):
            yield p
    for p in OTREE_XLSX:
        if p.exists():
            yield p


def apply_all(mapping, dry_run):
    pattern = build_pattern(mapping)
    tag = "[dry-run] " if dry_run else ""
    totals = {"text": 0, "xlsx": 0, "dirs": 0}

    print("\n--- text files (csv / jsonl / json) ---")
    per_group = {}
    for path in text_targets():
        n = rewrite_text_file(path, pattern, mapping, dry_run)
        if n:
            group = "data/prolific_data/*/*.json" if PROLIFIC_DATA in path.parents \
                else str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            per_group[group] = per_group.get(group, 0) + n
            totals["text"] += n
    for g, n in sorted(per_group.items()):
        print(f"  {tag}{n:>7} replacements  {g}")

    print("\n--- xlsx ---")
    for path in xlsx_targets():
        n = rewrite_xlsx(path, pattern, mapping, dry_run)
        totals["xlsx"] += n
        if n:
            print(f"  {tag}{n:>7} replacements  {path.relative_to(PROJECT_ROOT)}")

    print("\n--- directory names ---")
    if PROLIFIC_DATA.is_dir():
        for d in sorted(PROLIFIC_DATA.iterdir()):
            if d.is_dir() and d.name in mapping:
                target = d.with_name(mapping[d.name])
                if target.exists():
                    raise SystemExit(f"rename collision: {target}")
                if not dry_run:
                    d.rename(target)
                totals["dirs"] += 1
        print(f"  {tag}{totals['dirs']:>7} directories renamed under "
              f"{PROLIFIC_DATA.relative_to(PROJECT_ROOT)}")

    return totals


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def verify(originals):
    """Assert no original id survives anywhere."""
    print("\n=== VERIFICATION ===")
    leftovers = {}

    def check(label, text):
        hits = {t for t in re.findall(r"\b[0-9A-Za-z]{24}\b", text) if t in originals}
        if hits:
            leftovers[label] = len(hits)

    for path in text_targets():
        check(str(path.relative_to(PROJECT_ROOT)),
              path.read_text(encoding="utf-8", errors="ignore"))

    for path in xlsx_targets():
        wb = load_workbook(path, read_only=True, data_only=True)
        buf = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                for value in row:
                    if isinstance(value, str) and len(value) >= 24:
                        buf.append(value)
        wb.close()
        check(str(path.relative_to(PROJECT_ROOT)), "\n".join(buf))

    if PROLIFIC_DATA.is_dir():
        stale = [d.name for d in PROLIFIC_DATA.iterdir() if d.is_dir() and d.name in originals]
        if stale:
            leftovers[f"{PROLIFIC_DATA.name}/ (directory names)"] = len(stale)

    if leftovers:
        print("  FAILED -- original ids still present:")
        for k, v in sorted(leftovers.items()):
            print(f"    {v:>5} ids  {k}")
        return False
    print(f"  PASS -- none of the {len(originals)} original ids survive anywhere")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=lambda s: __import__("pathlib").Path(s), default=DEFAULT_MAP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    print("collecting original ids ...")
    originals = collect_original_ids()
    print(f"found {len(originals)} distinct participant ids")

    if args.verify_only:
        sys.exit(0 if verify(originals) else 1)

    mapping = load_or_create_map(args.map, originals, args.dry_run)
    totals = apply_all(mapping, args.dry_run)
    print(f"\ntotals: text={totals['text']} xlsx={totals['xlsx']} dirs={totals['dirs']}")

    if args.dry_run:
        print("\n[dry-run] nothing was modified.")
        return
    if not verify(originals):
        sys.exit(1)


if __name__ == "__main__":
    main()
