"""Diagnose a pilot run: did the token budget hold, and did the format hold?

    python -m scripts.pipeline.run_agent --config model_size/Qwen3.5-9B.json --pilot
    python -m scripts.pipeline.pilot_report results/model_size/pilot/*.xlsx

The two failure modes look identical in the parsed columns -- `llm_new_belief`
is empty either way -- so they are separated here by `finish_reason`:

    finish_reason == "length"   the model ran out of completion budget
    anything else, unparsed     the model ignored the JSON format instructions

Only the first is fixed by raising `max_tokens`; the second is a prompt or
model-capability problem, and for the smallest models in the size sweep it is a
result rather than a bug. Either way the number to carry into the write-up is
the per-model validity rate, which this prints.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# How close to the ceiling counts as "nearly truncated". A model whose longest
# pilot response used 95% of its budget will truncate on the full 391 personas
# even if nothing truncated in the pilot.
NEAR_LIMIT_FRACTION = 0.90

DIAGNOSTIC_COLUMNS = (
    "valid_output", "n_tries", "finish_reason",
    "completion_tokens", "prompt_tokens", "max_tokens_budget",
)


def _pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else float("nan")


def summarize(df: pd.DataFrame, label: str) -> dict:
    """One row of the pilot table, from one model's pilot workbook."""
    n = len(df)
    summary = {"model": label, "n": n}

    if "valid_output" in df.columns:
        valid = df["valid_output"].fillna(False).astype(bool).sum()
    else:
        # Workbooks written before the diagnostic columns existed.
        valid = df["llm_new_belief"].notna().sum() if "llm_new_belief" in df.columns else 0
    summary["valid"] = int(valid)
    summary["valid_pct"] = _pct(valid, n)

    # Truncation is counted over *attempts*, not final rows. A row whose first
    # attempt ran out of budget and whose retry succeeded still tells you the
    # budget is too small -- it just paid for the discovery twice.
    if "n_truncated" in df.columns and df["n_truncated"].notna().any():
        summary["truncated"] = int(pd.to_numeric(df["n_truncated"], errors="coerce").fillna(0).sum())
        summary["rows_truncated"] = int((pd.to_numeric(df["n_truncated"], errors="coerce").fillna(0) > 0).sum())
    elif "finish_reason" in df.columns:
        summary["truncated"] = int((df["finish_reason"] == "length").sum())
        summary["rows_truncated"] = summary["truncated"]
    else:
        summary["truncated"] = summary["rows_truncated"] = 0

    budget = None
    if "max_tokens_budget" in df.columns and df["max_tokens_budget"].notna().any():
        budget = int(df["max_tokens_budget"].dropna().max())
    summary["budget"] = budget

    # completion_tokens_max is the high-water mark across attempts; fall back to
    # the final attempt's count for workbooks written before it was recorded.
    token_col = ("completion_tokens_max"
                 if "completion_tokens_max" in df.columns
                 and df["completion_tokens_max"].notna().any()
                 else "completion_tokens")
    if token_col in df.columns and df[token_col].notna().any():
        tokens = pd.to_numeric(df[token_col], errors="coerce").dropna()
        summary["tok_median"] = int(tokens.median())
        summary["tok_max"] = int(tokens.max())
        summary["budget_used_pct"] = _pct(tokens.max(), budget) if budget else float("nan")
        summary["near_limit"] = int((tokens >= NEAR_LIMIT_FRACTION * budget).sum()) if budget else 0
    else:
        summary["tok_median"] = summary["tok_max"] = None
        summary["budget_used_pct"] = float("nan")
        summary["near_limit"] = 0

    # Rows whose answer came from the no-thinking fallback. These are valid
    # output, but they were not produced the same way as the rest, so they are
    # counted separately rather than folded into the validity rate.
    if "thinking_disabled" in df.columns:
        summary["no_thinking"] = int(df["thinking_disabled"].fillna(False).astype(bool).sum())
    else:
        summary["no_thinking"] = 0

    if "n_tries" in df.columns and df["n_tries"].notna().any():
        summary["mean_tries"] = float(pd.to_numeric(df["n_tries"], errors="coerce").mean())
    else:
        summary["mean_tries"] = float("nan")

    # How much of the generation was thinking rather than answer. A model using
    # a quarter of its budget almost entirely on thinking behaves very
    # differently from one writing a long answer, and only the first is at risk
    # of running out of room as prompts get harder.
    if "reasoning_tokens" in df.columns and df["reasoning_tokens"].notna().any():
        reasoning = pd.to_numeric(df["reasoning_tokens"], errors="coerce")
        completion = pd.to_numeric(df.get("completion_tokens"), errors="coerce")
        both = reasoning.notna() & completion.notna() & (completion > 0)
        summary["think_pct"] = (_pct(reasoning[both].sum(), completion[both].sum())
                                if both.any() else float("nan"))
    else:
        summary["think_pct"] = float("nan")

    return summary


def verdict(row: dict) -> str:
    """What to do about this model before committing to the full run."""
    if row["n"] == 0:
        return "NO DATA - the run produced no rows"
    if row["truncated"]:
        # A row that only parsed once thinking was switched off is not evidence
        # that the budget was too small -- the fallback exists because for these
        # models it is not. Say so, rather than recommending a bigger ceiling.
        if row.get("no_thinking"):
            return (f"NO-THINKING FALLBACK - {row['truncated']} attempt(s) across "
                    f"{row['rows_truncated']}/{row['n']} rows hit the ceiling; "
                    f"{row['no_thinking']}/{row['n']} row(s) answered with thinking off")
        recovered = "" if row["valid_pct"] < 100 else " (retries recovered, but they cost a call each)"
        return (f"RAISE max_tokens - {row['truncated']} attempt(s) across "
                f"{row['rows_truncated']}/{row['n']} rows hit the ceiling{recovered}")
    if row["near_limit"]:
        return (f"RAISE max_tokens - {row['near_limit']}/{row['n']} used over "
                f"{NEAR_LIMIT_FRACTION:.0%} of it")
    if row["valid_pct"] < 100:
        return f"FORMAT - {row['n'] - row['valid']}/{row['n']} unparseable, budget was not the limit"
    return "OK"


def load_workbook(path: Path) -> tuple[str, pd.DataFrame]:
    df = pd.read_excel(path)
    if "model" in df.columns and df["model"].notna().any():
        label = str(df["model"].dropna().iloc[0])
    else:
        label = path.stem
    return label, df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workbooks", nargs="+",
                        help="pilot .xlsx files (globs are expanded on Windows too)")
    parser.add_argument("--show-failures", type=int, default=0, metavar="N",
                        help="print the raw response of up to N failed rows per model")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.workbooks:
        expanded = [Path(p) for p in glob.glob(pattern)]
        paths.extend(expanded if expanded else [Path(pattern)])

    missing = [p for p in paths if not p.exists()]
    if missing:
        print("error: no such file(s): " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 1

    rows = []
    failures = {}
    for path in sorted(paths):
        label, df = load_workbook(path)
        row = summarize(df, label)
        row["verdict"] = verdict(row)
        rows.append(row)
        if args.show_failures and "valid_output" in df.columns:
            bad = df[~df["valid_output"].fillna(False).astype(bool)]
            failures[label] = bad.head(args.show_failures)

    table = pd.DataFrame(rows)
    display = table.assign(
        valid=lambda d: d.apply(lambda r: f"{r['valid']}/{r['n']} ({r['valid_pct']:.0f}%)", axis=1),
        tokens=lambda d: d.apply(
            lambda r: "-" if r["tok_max"] is None
            else f"{r['tok_median']}/{r['tok_max']} of {r['budget']} ({r['budget_used_pct']:.0f}%)",
            axis=1),
        tries=lambda d: d["mean_tries"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
        think=lambda d: d["think_pct"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}%"),
    )[["model", "n", "valid", "tokens", "think", "tries", "truncated",
       "no_thinking", "verdict"]]
    display.columns = ["model", "n", "valid", "tokens med/max of budget",
                       "think", "tries", "trunc", "no-think", "verdict"]

    print(display.to_string(index=False))
    print("\n'tokens' is completion tokens (thinking included), high-water mark "
          "across attempts, against max_tokens.")
    print("'trunc' counts attempts that ended with finish_reason == 'length', "
          "including ones a retry recovered from.")
    print("'think' is the share of generated tokens that was thinking rather "
          "than answer.")
    print("'no-think' counts rows whose answer came from the no-thinking "
          "fallback, after two truncated attempts in a row.")

    for label, bad in failures.items():
        if bad.empty:
            continue
        print(f"\n--- {label}: first {len(bad)} failed response(s) ---")
        for _, r in bad.iterrows():
            print(f"  [{r.get('persona_id')} / {r.get('topic')}] "
                  f"finish_reason={r.get('finish_reason')} "
                  f"completion_tokens={r.get('completion_tokens')} "
                  f"reasoning_chars={r.get('reasoning_chars')}")
            raw = r.get("raw_response")
            raw = str(raw) if isinstance(raw, str) else ""
            if raw:
                print(f"    content: {raw[:600]!r}")
            else:
                print("    content: (empty -- the whole budget went to thinking)")
            # The excerpt is head+tail: identical text at both ends means the
            # model was looping, rather than reasoning at length.
            excerpt = r.get("reasoning_excerpt")
            if isinstance(excerpt, str) and excerpt:
                print("    thinking (head/tail):")
                for line in excerpt.splitlines():
                    print(f"      {line}")

    # Non-zero exit when any model needs attention, so the SLURM pilot job fails
    # visibly instead of printing a warning into a log nobody reads.
    return 1 if any(r["verdict"] != "OK" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
