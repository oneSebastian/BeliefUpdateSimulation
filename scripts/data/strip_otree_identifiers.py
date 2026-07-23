"""Remove oTree-internal identifying columns from the raw participant exports.

Dropped:
  participant.code              per-participant token, unique 391/391, no analytic use
  session.code                  single value across all participants -- zero information
  participant.time_started_utc  near-unique (387/391) start timestamp; the strongest
                                external linkage vector, since Prolific holds start
                                and submission times for every participant

Demographic fields are deliberately retained.

These exports store an entire CSV row inside a single spreadsheet cell. Six rows
per file carry one extra field because a quoted free-text answer spilled during
the original paste. The three target columns sit at positions 1, 8 and 13 --
well before the free-text fields -- so they are removed *positionally* and every
other field is passed through untouched. Nothing is re-normalised, so the export
stays faithful to the original rather than silently rewriting the malformed rows.

    python -m scripts.data.strip_otree_identifiers --dry-run
    python -m scripts.data.strip_otree_identifiers
"""

import argparse
import csv
import io

import pandas as pd
from openpyxl import Workbook, load_workbook

from belief_update_sim.config import DATA_DIR

TARGETS = [DATA_DIR / "400_participant_otree.xlsx",
           DATA_DIR / "cleaned_391_participant_otree.xlsx"]

DROP_COLUMNS = ("participant.code", "session.code", "participant.time_started_utc")
EXCEL_CELL_LIMIT = 32767


def read_rows(path):
    """Reconstruct the embedded CSV: one spreadsheet row -> one CSV record."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    lines = []
    for row in raw.values:
        cells = ["" if pd.isna(v) else str(v) for v in row]
        while cells and cells[-1] == "":
            cells.pop()
        lines.append(",".join(cells))
    return list(csv.reader(io.StringIO("\n".join(lines))))


def drop_positions(header):
    pos = []
    for name in DROP_COLUMNS:
        hits = [i for i, h in enumerate(header) if h.strip().endswith(name)]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one {name!r} column, found {len(hits)}")
        pos.append(hits[0])
    return sorted(pos, reverse=True)


def serialize(record):
    buf = io.StringIO()
    csv.writer(buf, lineterminator="").writerow(record)
    return buf.getvalue()


def process(path, dry_run):
    rows = read_rows(path)
    header, data = rows[0], rows[1:]
    pos = drop_positions(header)
    names = [header[i] for i in sorted(pos)]

    print(f"\n{path.name}")
    print(f"  rows={len(data)}  header fields={len(header)}")
    print(f"  dropping positions {sorted(pos)} -> {[n.split('.')[-1] for n in names]}")

    new_rows = []
    for rec in rows:
        rec = list(rec)
        for i in pos:                      # descending, so indices stay valid
            if i < len(rec):
                del rec[i]
        new_rows.append(rec)

    longest = max(len(serialize(r)) for r in new_rows)
    print(f"  longest serialized row: {longest} chars "
          f"({'OK' if longest < EXCEL_CELL_LIMIT else 'EXCEEDS EXCEL LIMIT'})")
    if longest >= EXCEL_CELL_LIMIT:
        raise SystemExit("a row would exceed Excel's per-cell limit; aborting")

    # sanity: the retained label column must survive untouched
    new_header = new_rows[0]
    lab = [i for i, h in enumerate(new_header) if h.strip().endswith("participant.label")]
    if lab:
        vals = [r[lab[0]] for r in new_rows[1:] if len(r) > lab[0]]
        print(f"  participant.label retained: {len(set(vals))} distinct, e.g. {vals[:3]}")
    for name in DROP_COLUMNS:
        assert not any(h.strip().endswith(name) for h in new_header), name

    if dry_run:
        print("  [dry-run] not written")
        return

    wb = Workbook()
    ws = wb.active
    for rec in new_rows:
        ws.append([serialize(rec)])
    wb.save(path)
    print(f"  written ({len(new_rows)} cells)")


def verify():
    print("\n=== VERIFICATION ===")
    ok = True
    for path in TARGETS:
        rows = read_rows(path)
        header = rows[0]
        present = [n for n in DROP_COLUMNS
                   if any(h.strip().endswith(n) for h in header)]
        lab = any(h.strip().endswith("participant.label") for h in header)
        print(f"  {path.name}: fields={len(header)} rows={len(rows)-1} "
              f"| dropped-cols still present={present or 'NONE'} "
              f"| participant.label retained={lab}")
        ok &= not present and lab
    print("  PASS" if ok else "  FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for path in TARGETS:
        if path.exists():
            process(path, args.dry_run)
        else:
            print(f"\n{path.name}: not found, skipped")

    if not args.dry_run:
        raise SystemExit(0 if verify() else 1)


if __name__ == "__main__":
    main()
