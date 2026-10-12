"""Model-size sweep: does a bigger model simulate the humans more faithfully?

This module *computes* and writes a tidy CSV. It does not render anything for
the paper -- :mod:`scripts.stats.model_size_table` reads that CSV and emits the
LaTeX. The split is deliberate: the bootstrap and the permutation tests take
minutes, and reformatting a table should not pay for them again.

    python -m scripts.stats.model_size              # -> model_size.{csv,json,txt}
    python -m scripts.stats.model_size_table        # -> model_size.tex, instantly

Per model it reports:

    Delta  mean(human - model) final stance, the paired difference the
           permutation test tests. Signed: > 0 means the humans end more
           pro-topic than the model. A model can score Delta ~ 0 by being
           wrong in both directions equally, so it is not a quality measure
    MAE    mean |human - model| final stance. This is the quality measure and
           the one the ranking uses -- it cannot cancel

with a permutation p-value for Delta, Holm-corrected across the sweep.

Almost no statistics are computed here. The scores, bootstrap CIs, ranks and
P(best) come from :mod:`scripts.stats.model_ranking`, whose ``compute`` already
takes an arbitrary frame and model list; only its ``load`` is specific to the
paper's API models. The p-values come from
:func:`belief_update_sim.permutation_test.fast_permutation_test`, the same
function :mod:`scripts.stats.effect_size_report` uses, so the values agree with
the rest of the reporting.

COMPLETE CASES. Every source is scored on the persona x topic cases where
*every* model in the sweep parsed, which is what makes MAE comparable at all --
otherwise each model is graded on a different, self-selected set of rows. The
cost is real and not missing-at-random: the small models fail on the cases they
find hard, so scoring them only on what they answered flatters them. That is
why "valid n (%)" is a column rather than a footnote. Read every row as
conditional on it.

The registry is read from ``configs/model_size/*.json`` -- the same twelve
configs the sweep is submitted from -- so the analysis cannot drift from what
was run. A model whose results file is short of the human frame is a run still
in flight; it is excluded by name rather than silently folded in, because
including a partial model shrinks the complete-case set for everyone.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from belief_update_sim.config import (CONFIGS_DIR, PROJECT_ROOT, STATS_OUTPUT_DIR,
                                      ensure_output_dirs)
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data
from belief_update_sim.permutation_test import (PERMUTATION_RESAMPLES,
                                                fast_permutation_test)

from scripts.stats import model_ranking as mr

MODEL_SIZE_CONFIG_DIR = CONFIGS_DIR / "model_size"
CSV_PATH = STATS_OUTPUT_DIR / "model_size.csv"
JSON_PATH = STATS_OUTPUT_DIR / "model_size.json"

# Models to leave out of the sweep, by name. Empty: all twelve have finished.
# Qwen3.5-A10B sat here while its run was in flight, because scoring a partial
# model shrinks the complete-case set for every other one -- it was at 861 of
# 1173 cases, which would have cut the intersection from 970 to 715. `load`
# enforces the same rule independently, so a run that is still going cannot be
# folded in by forgetting to list it here.
EXCLUDED = ()

ALPHA = 0.05            # family-wise, Holm across the models in the sweep

# --------------------------------------------------------------------------
# Model size
#
# Names in this sweep encode size in three different ways, and conflating them
# would order the table wrongly:
#
#   dense        "Qwen3.5-9B"          9B stored, 9B active per token
#   MoE          "Qwen3.5-35B-A3B"     35B stored, 3B active per token
#   Gemma "E"    "gemma-4-E2B-it"      ~5B stored, 2B effective
#
# The last is not sparsity. Gemma's E-naming states *effective* parameters:
# Per-Layer Embeddings are not resident in accelerator memory, so the
# checkpoint is far larger than the number in the name. That is confirmed
# below, not assumed -- MEASURED_CHECKPOINT_GIB comes from the sweep's own vLLM
# startup lines ("Checkpoint size: X GiB") and every dense and MoE name matches
# its checkpoint within ~5%, while E2B and E4B are off by 2.6x and 2.0x.
#
# `active` is the column the table sorts on, because it is compute per token
# and so the honest x-axis for "does more model help". `total` is carried
# alongside, because capacity is not nothing.
# --------------------------------------------------------------------------

_SIZE_RE = re.compile(r"(E?)(\d+(?:\.\d+)?)B")
_ACTIVE_RE = re.compile(r"-A(\d+(?:\.\d+)?)B")

BYTES_PER_PARAM = 2          # every model in the sweep is served at bf16

# Read from logs/vllm_<model>_<jobid>.out. logs/ is gitignored, so the
# measurement is recorded here rather than re-read at runtime.
MEASURED_CHECKPOINT_GIB = {
    "Qwen3.5-0.8B": 1.63,
    "Qwen3.5-2B": 4.24,
    "Qwen3.5-4B": 8.68,
    "Qwen3.5-9B": 17.98,
    "Qwen3.5-27B": 51.75,
    "Qwen3.5-35B-A3B": 66.97,
    "Qwen3.5-122B-A10B": 232.99,
    "gemma-4-E2B-it": 9.54,
    "gemma-4-E4B-it": 14.89,
    "gemma-4-12B-it": 22.28,
    "gemma-4-26B-A4B-it": 48.07,
    "gemma-4-31B-it": 58.25,
}


def measured_params_b(model):
    """Parameters in billions, from the served checkpoint's size. None if the
    model was not in the logs these were read from."""
    gib = MEASURED_CHECKPOINT_GIB.get(model)
    if gib is None:
        return None
    return gib * 2 ** 30 / BYTES_PER_PARAM / 1e9


def parse_size(model):
    """(total_b, active_b) as the model's own name states them.

    For a Gemma "E" name the number is the effective size, so it is the active
    figure and the total comes from the checkpoint. For an MoE name the number
    is the total and "-A<n>B" is the active figure. For anything else the two
    are the same.
    """
    match = _SIZE_RE.search(model)
    if match is None:
        raise ValueError(f"{model!r} has no '<number>B' token to size it by")
    effective, stated = match.group(1) == "E", float(match.group(2))

    if effective:
        measured = measured_params_b(model)
        return (measured if measured is not None else stated), stated

    active = _ACTIVE_RE.search(model)
    return stated, float(active.group(1)) if active else stated


def family(model):
    """The name up to its size token: 'Qwen3.5-9B' -> 'Qwen3.5'."""
    parts = model.split("-")
    for i, part in enumerate(parts):
        if _SIZE_RE.fullmatch(part):
            if i == 0:
                break
            return "-".join(parts[:i])
    raise ValueError(f"{model!r} has no size token to split the family on")


def size_key(model):
    """Sort key along the size axis: active parameters, then total, then name.

    Active rather than total, so a sparse model sits where its per-token
    compute puts it. The name is the final tie-break, so models of equal size
    keep a stable order rather than depending on directory listing order.
    """
    total, active = parse_size(model)
    return active, total, model


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def sweep_entries(config_dir=MODEL_SIZE_CONFIG_DIR, exclude=EXCLUDED):
    """{model: {path, max_tries}}, smallest first, read from the sweep's configs.

    The configs are the single source of truth for what the sweep is: they are
    what submit_model.sh runs and what serving_params validates. A second
    hand-maintained list here would drift from them. `max_tries` comes along
    because the retry column counts runs that used *every* retry they were
    allowed, which is only meaningful against the budget the run actually had.
    """
    entries = {}
    for path in sorted(config_dir.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8-sig"))[0]
        name = entry["model"]
        if name in exclude:
            continue
        entries[name] = {"path": PROJECT_ROOT / entry["paths"]["output_excel"],
                         "max_tries": int(entry["max_tries"])}
    if not entries:
        raise SystemExit(f"no model-size configs found in {config_dir}")
    return {name: entries[name] for name in sorted(entries, key=size_key)}


def load(models):
    """(frame, dropped, valid): the complete-case frame and the validity counts.

    `dropped` is what assemble() reports -- per model, the cases it did not
    parse. `valid` turns that into (n, share) against the human frame, which is
    the column the table carries.
    """
    human = load_human_normalized_data()

    frames = {}
    for name, path in models.items():
        if not path.exists():
            raise SystemExit(f"missing results file for {name}: {path}")
        df = load_normalized_data(path)
        if len(df) < len(human):
            raise SystemExit(
                f"{name}: {len(df)} rows against the humans' {len(human)}. That is a "
                f"run still in flight; add it to EXCLUDED until it finishes rather "
                f"than scoring the sweep on the cases it happens to have reached."
            )
        frames[name] = df

    frame, dropped = mr.assemble(human, frames)
    total = len(human)
    valid = {name: {"n": total - dropped[name], "total": total,
                    "share": (total - dropped[name]) / total}
             for name in frames}
    return frame, dropped, valid


def retry_rates(entries):
    """{model: {n, total, share, max_tries}}: how often the agent needed every
    retry it was allowed.

    `n_tries == max_tries` is the run's own record of having struggled. It is a
    different signal from validity: a row can exhaust its retries and still
    parse on the last one, so it counts here and not as a failure. Read over
    all of a model's rows, like validity and unlike every score in the table,
    because it describes the run rather than the comparison.
    """
    out = {}
    for name, entry in entries.items():
        tries = pd.read_excel(entry["path"], usecols=["n_tries"])["n_tries"]
        at_ceiling = int((tries == entry["max_tries"]).sum())
        out[name] = {"n": at_ceiling, "total": int(len(tries)),
                     "share": at_ceiling / len(tries),
                     "max_tries": entry["max_tries"]}
    return out


# --------------------------------------------------------------------------
# the permutation test, on the complete-case frame
# --------------------------------------------------------------------------

def permutation_rows(frame, sources, alpha=ALPHA, n_resamples=PERMUTATION_RESAMPLES):
    """{source: {delta, p, p_holm, significant}} on the *assembled* frame.

    Deliberately not permutation_test_on_paths: that re-merges from the files
    and would test each model on its own rows, so the p-value in a table row
    would describe a different sample than the MAE beside it. The frame is
    already complete-case, so _reject_nulls has nothing to refuse.
    """
    sources = list(sources)
    deltas, pvals = [], []
    for name in sources:
        delta, p, _ = fast_permutation_test(frame["human"], frame[name], n_resamples)
        deltas.append(float(delta))
        pvals.append(float(p))

    reject, p_holm, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return {name: {"delta": deltas[i], "p": pvals[i], "p_holm": float(p_holm[i]),
                   "significant": bool(reject[i])}
            for i, name in enumerate(sources)}


def compute(frame, models, valid, retries, n_boot=mr.N_BOOT, n_perm=mr.N_PERM,
            seed=mr.SEED, alpha=ALPHA, n_resamples=PERMUTATION_RESAMPLES):
    """model_ranking's scores for this sweep, plus p-values, validity, retries.

    `n_perm` is model_ranking's budget for the pairwise MAE tests; `n_resamples`
    is this module's for the Delta p-values. They are separate tests, so they
    get separate knobs -- conflating them is how a cheap test run silently
    changes a published p.
    """
    llms = list(models)
    r = mr.compute(frame, llms, n_boot, n_perm, seed, alpha)
    sources = llms + r["baselines"]
    r["perm"] = permutation_rows(frame, sources, alpha, n_resamples)
    r["n_resamples"] = n_resamples
    r["valid"] = valid
    r["retries"] = retries
    r["size_order"] = llms                  # sweep_entries already sorted these
    r["excluded"] = list(EXCLUDED)
    return r


# --------------------------------------------------------------------------
# the tidy table -- the handover to the renderer
# --------------------------------------------------------------------------

def to_frame(r):
    """One row per source, everything a renderer needs, and nothing derived.

    The run-level facts (n, replicates, seed, what was excluded) are repeated on
    every row so the CSV is self-contained: the renderer opens one file and can
    write the caption from it, with no chance of pairing a table with the wrong
    run's metadata.
    """
    rows = []
    for name in r["size_order"] + r["baselines"]:
        baseline = name not in r["size_order"]
        mae, perm = r["metrics"]["mae"][name], r["perm"][name]
        # `bias` is model_ranking's name for the same statistic the permutation
        # test tests: mean(human - model). Its CI belongs beside Delta, not
        # beside MAE -- they are intervals for different quantities.
        bias = r["metrics"]["bias"][name]
        v, t = r["valid"].get(name), r["retries"].get(name)
        total, active = (None, None) if baseline else parse_size(name)
        rows.append({
            "source": name,
            "is_baseline": baseline,
            "family": None if baseline else family(name),
            "params_total_b": total,
            "params_active_b": active,
            "params_measured_b": None if baseline else measured_params_b(name),
            "is_sparse": None if baseline else (active != total),
            "valid_n": v["n"] if v else None,
            "valid_total": v["total"] if v else None,
            "valid_share": v["share"] if v else None,
            "tries_n": t["n"] if t else None,
            "tries_share": t["share"] if t else None,
            "max_tries": t["max_tries"] if t else None,
            "mae": mae["estimate"],
            "mae_lo": mae["ci"][0],
            "mae_hi": mae["ci"][1],
            "rank": mae.get("rank"),
            "delta": perm["delta"],
            "delta_lo": bias["ci"][0],
            "delta_hi": bias["ci"][1],
            "p": perm["p"],
            "p_holm": perm["p_holm"],
            "significant": perm["significant"],
            "significant_raw": perm["p"] < r["alpha"],
            "tier": r["letters"].get(name),
            # run-level, repeated so the file stands alone
            "n_obs": r["n_obs"],
            "n_participants": r["n_participants"],
            "n_boot": r["n_boot"],
            "n_perm": r["n_perm"],
            "seed": r["seed"],
            "alpha": r["alpha"],
            "n_resamples": r.get("n_resamples", PERMUTATION_RESAMPLES),
            "excluded": ";".join(r["excluded"]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# a plain-text view, for reading at the terminal
# --------------------------------------------------------------------------

PLACES = 4
FLOOR = 10 ** -PLACES

RULE = "=" * 118
LINE = "-" * 118


def p_text(p):
    """Four decimal places, with a threshold where that would round to zero.

    Printing 0.0000 would claim the null was excluded exactly, which no
    resampling test can do -- and fast_permutation_test really can return 0.0
    (its resolution is 1/1e6). '< 0.0001' is the honest reading of both that
    and any p below the fourth place.
    """
    return f"< {FLOOR:.4f}" if p < FLOOR else format(p, f".{PLACES}f")


def render(r):
    lines = []
    out = lines.append
    table = to_frame(r)

    out(RULE)
    out("Model-size sweep: paired stance difference and absolute error")
    out(f"n = {r['n_obs']} complete cases from {r['n_participants']} participants "
        f"(every model parsed); bootstrap {r['n_boot']}x, seed {r['seed']}")
    if r["excluded"]:
        out(f"excluded, run not finished: {', '.join(r['excluded'])}")
    out(RULE)
    out(f"{'model':<24}{'total':>7}{'active':>8}{'valid%':>8}{'tries%':>8}"
        f"{'MAE':>9}{'MAE 95% CI':>18}{'rank':>6}"
        f"{'Delta':>9}{'Delta 95% CI':>20}{'p':>11}{'p_holm':>11}")
    out(LINE)
    for _, row in table.iterrows():
        def num(value, spec):
            return "-" if pd.isna(value) else format(value, spec)
        valid = row["valid_share"] * 100 if pd.notna(row["valid_share"]) else None
        tries = row["tries_share"] * 100 if pd.notna(row["tries_share"]) else None
        out(f"{row['source']:<24}"
            f"{num(row['params_total_b'], '.1f'):>7}"
            f"{num(row['params_active_b'], '.1f'):>8}"
            f"{num(valid, '.1f'):>8}{num(tries, '.1f'):>8}"
            f"{row['mae']:>9.4f}"
            f"{f'[{row.mae_lo:.4f}, {row.mae_hi:.4f}]':>18}"
            f"{num(row['rank'], '.0f'):>6}{row['delta']:>+9.4f}"
            f"{f'[{row.delta_lo:+.4f}, {row.delta_hi:+.4f}]':>20}"
            f"{p_text(row['p']):>11}{p_text(row['p_holm']):>11}")
    out(LINE)
    out("total/active are billions of parameters; they differ for MoE models and "
        "for Gemma's")
    out("effective-parameter (E) names. The table sorts on active -- compute per "
        "token.")
    out("Delta = mean(human - model) final stance; > 0 means the humans end more "
        "pro-topic.")
    out("p is the permutation test's own two-sided value, uncorrected. It tests "
        "Delta, the")
    out("paired SIGNED difference, so it goes with the Delta CI; the MAE CI is an "
        "interval for a")
    out(f"different quantity. p_holm is p times its Holm rank factor, capped at 1, "
        f"across the")
    out(f"{len(table)} sources at alpha = {r['alpha']} -- a different number, not a "
        f"restatement of p.")
    out("Each CI is a participant-level bootstrap percentile interval. Its width "
        "comes from")
    out(f"variation between the {r['n_participants']} participants, not from the "
        f"{r['n_boot']} replicates: more")
    out("replicates pin the endpoints down more precisely, they do not narrow the "
        "interval.")
    out("valid% is the cases a model parsed and tries% the cases that used every "
        "retry allowed;")
    out("both are over ALL of a model's rows, before the complete-case "
        "intersection. Every score")
    out("above is computed on the intersection, so a low validity rate means that "
        "model's numbers")
    out("rest on the subset it could answer -- which is not a random subset.")
    out("")
    out("MAE tiers (sources sharing a letter do not differ significantly): "
        + ", ".join(f"{n} {r['letters'][n]}" for n in r["order"]))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-boot", type=int, default=mr.N_BOOT)
    parser.add_argument("--n-perm", type=int, default=mr.N_PERM)
    parser.add_argument("--seed", type=int, default=mr.SEED)
    parser.add_argument(
        "--from-json", metavar="PATH", nargs="?", const=str(JSON_PATH),
        help="rebuild the CSV and the text report from a previous run's JSON "
             "instead of recomputing. The JSON holds everything the bootstrap "
             "and the permutation tests produced, so adding a column to the CSV "
             "does not have to cost minutes again.")
    args = parser.parse_args()

    if args.from_json:
        r = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
        print(f"rebuilding from {args.from_json} -- nothing is recomputed")
    else:
        entries = sweep_entries()
        models = {name: e["path"] for name, e in entries.items()}
        frame, dropped, valid = load(models)
        r = compute(frame, models, valid, retry_rates(entries),
                    args.n_boot, args.n_perm, args.seed)
        r["dropped"] = dropped

    ensure_output_dirs()
    text = render(r)
    print(text)
    to_frame(r).to_csv(CSV_PATH, index=False)
    (STATS_OUTPUT_DIR / "model_size.txt").write_text(text, encoding="utf-8")
    written = [CSV_PATH, STATS_OUTPUT_DIR / "model_size.txt"]
    # Rebuilding from the JSON must not rewrite it: it is the source being read,
    # and round-tripping it through json.dumps would only risk damaging it.
    if not args.from_json:
        JSON_PATH.write_text(json.dumps(r, indent=2), encoding="utf-8")
        written.append(JSON_PATH)
    for path in written:
        print(f"wrote {path}")
    print("\nrender the paper table with:  python -m scripts.stats.model_size_table")


if __name__ == "__main__":
    main()
