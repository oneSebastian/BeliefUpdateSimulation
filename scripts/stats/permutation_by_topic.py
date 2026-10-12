"""Paired permutation tests split by topic and by message package, as LaTeX.

For every design cell x model: the mean paired final-stance difference (human
- matched LLM, Delta), its two-sided sign-flip permutation p, and the mean
absolute paired difference (MAE). Delta can cancel opposing errors to near
zero; MAE cannot, which is why it is reported beside it. The p-value tests
Delta only -- MAE is descriptive.

Two tables come out of one run:

    permutation_by_topic.tex     3 topics x 6 models                (18 tests)
    permutation_by_package.tex   9 topic x package cells x 6 models (54 tests)

Bonferroni is applied within each table, over that table's tests.

    python -m scripts.stats.permutation_by_topic             # a few min, 72 tests
    python -m scripts.stats.permutation_by_topic --rebuild   # instant, from JSON

A full run writes the numbers to permutation_by_topic.json; ``--rebuild``
re-renders both tables from it, so reformatting does not pay for the
permutations again.
"""

import argparse
import json

import pandas as pd

from belief_update_sim.comment_ranks import MODELS
from belief_update_sim.config import RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data
from belief_update_sim.grouping import PACKAGES
from belief_update_sim.permutation_test import PERMUTATION_RESAMPLES, fast_permutation_test

JSON_PATH = STATS_OUTPUT_DIR / "permutation_by_topic.json"
TOPIC_TEX_PATH = STATS_OUTPUT_DIR / "permutation_by_topic.tex"
PACKAGE_TEX_PATH = STATS_OUTPUT_DIR / "permutation_by_package.tex"

# topic order and headings in both tables
TOPIC_LABELS = {"UBI": "UBI",
                "penalty": "Penalty Shootout",
                "weight_loss": "Weight Loss Drugs"}

# row order and labels (keys are the MODELS display names)
MODEL_LABELS = {"Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
                "GPT-5.2": "GPT-5.2",
                "Claude-Opus-4.6": "Claude-Opus-4.6",
                "Qwen-3-32B": "Qwen3-32B",
                "Gemini-3-Flash-Preview": "Gemini-3-Flash-Preview",
                "GPT-5-Mini": "GPT-5-Mini"}

FLOOR = 1e-4    # p below this is printed as '<0.0001'


def paired(human, model, name=""):
    """(human, model) post-stances for every participant-topic both answered.

    A pair whose comment package differs between the two sources is an error:
    it would mean the human and the model did not read the same comments.
    Pairs where either side has no parsed stance are dropped, since the
    permutation test refuses NaN.
    """
    merged = pd.merge(human, model, on=["persona_id", "topic"],
                      suffixes=("_h", "_m"), validate="one_to_one")
    mismatched = merged["package_h"] != merged["package_m"]
    if mismatched.any():
        raise ValueError(
            f"{name}: {int(mismatched.sum())} participant-topics were shown a "
            "different comment package by the human study and the model run."
        )
    both = merged[["new_belief_h", "new_belief_m"]].dropna()
    return both["new_belief_h"], both["new_belief_m"]


def cell(s_human, s_model, n_resamples=PERMUTATION_RESAMPLES):
    """Delta, MAE and permutation p for one design cell x model."""
    diff = s_human - s_model
    _, p, _ = fast_permutation_test(s_human, s_model, n_resamples)
    return {"n": int(diff.size), "delta": float(diff.mean()),
            "mae": float(diff.abs().mean()), "p": float(p)}


def compute(n_resamples=PERMUTATION_RESAMPLES):
    """{"topic": {topic: {model: cell}},
        "package": {topic: {package: {model: cell}}}}

    Only the human side is subset; each model frame is merged whole, so a
    package disagreement surfaces in `paired` instead of being filtered away.
    """
    human = load_human_normalized_data()
    models = {name: load_normalized_data(RESULTS_DIR / MODELS[name])
              for name in MODEL_LABELS}

    def run(h, label):
        if h.empty:
            raise ValueError(f"no human rows for {label}")
        row = {}
        for name, model in models.items():
            row[name] = cell(*paired(h, model, name), n_resamples)
            print(f"{label:<24} {name:<24} n={row[name]['n']}", flush=True)
        return row

    by_topic, by_package = {}, {}
    for topic in TOPIC_LABELS:
        h = human[human["topic"] == topic]
        by_topic[topic] = run(h, topic)
        by_package[topic] = {pkg: run(h[h["package"] == pkg], f"{topic} {pkg}")
                             for pkg in PACKAGES}
    return {"n_resamples": n_resamples, "topic": by_topic, "package": by_package}


def num(value):
    """Four decimals, without a '-0.0000' for a value that rounds to zero."""
    value = round(value, 4)
    return f"{0.0 if value == 0 else value:.4f}"


def p_cell(p, alpha):
    text = "$<$0.0001" if p < FLOOR else f"{p:.4f}"
    return r"\textbf{" + text + "}" if p < alpha else text


def triple(c, alpha):
    """Delta, p, MAE -- p beside the Delta it tests, MAE to its right."""
    return [num(c["delta"]), p_cell(c["p"], alpha), num(c["mae"])]


def caption(split, within, n_resamples, n_tests):
    resamples = f"{n_resamples:,}".replace(",", "{,}")
    return (
        r"\caption{Paired permutation test results split by " + split + ". "
        r"$\Delta$ denotes the mean paired final stance difference (human "
        r"post-stance minus matched LLM post-stance) within each " + within
        + r"; MAE is the mean absolute paired difference, which unlike "
        r"$\Delta$ cannot cancel opposing errors. $p$-values test "
        r"$\Delta = 0$, are based on " + resamples + r" sign-flip resamples "
        r"and give the probability, under the null hypothesis, of observing a "
        r"mean paired difference at least as extreme as the one observed; MAE "
        r"is descriptive. Bold $p$-values are significant after Bonferroni "
        r"correction across all " + str(n_tests) + r" tests ($p < 0.05/"
        + str(n_tests) + r"$).}"
    )


def table(caption_text, label, group_labels, body):
    """The booktabs frame shared by both tables: l + one rrr block per group."""
    lines = [
        r"% Generated by scripts.stats.permutation_by_topic -- do not edit by hand.",
        r"% Regenerate:  python -m scripts.stats.permutation_by_topic --rebuild",
        r"% Requires \usepackage{booktabs}",
        r"\begin{table}[ht]",
        r"\centering",
        caption_text,
        r"\label{" + label + "}",
        r"\small",
        r"\begin{tabular}{l" + " rrr" * len(group_labels) + "}",
        r"\toprule",
        " & " + " & ".join(r"\multicolumn{3}{c}{" + g + "}"
                           for g in group_labels) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
                 for i in range(len(group_labels))),
        "Model & " + " & ".join([r"$\Delta$ & $p$ & MAE"] * len(group_labels))
        + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _models(cells):
    """MODEL_LABELS order, restricted to models present in every cell."""
    return [m for m in MODEL_LABELS if all(m in c for c in cells)]


def _row(name, width, cells):
    return f"{MODEL_LABELS[name]:<{width}} & " + " & ".join(cells) + r" \\"


def render_topic(payload):
    results = payload["topic"]
    topics = [t for t in TOPIC_LABELS if t in results]
    models = _models([results[t] for t in topics])
    n_tests = len(topics) * len(models)
    alpha = 0.05 / n_tests
    width = max(len(MODEL_LABELS[m]) for m in models)

    body = [_row(name, width,
                 [x for t in topics for x in triple(results[t][name], alpha)])
            for name in models]
    return table(caption("discussion topic", "topic", payload["n_resamples"], n_tests),
                 "tab:si_permutation_topic", [TOPIC_LABELS[t] for t in topics], body)


def render_package(payload):
    """Models down the side under a heading row per topic, packages across.

    The other way round -- cells down the side, models across -- needs
    6 models x 3 values = 18 number columns and only fits scaled down. This
    way it has the 9 number columns of the topic table.
    """
    results = payload["package"]
    topics = [t for t in TOPIC_LABELS if t in results]
    models = _models([results[t][p] for t in topics for p in PACKAGES])
    n_tests = len(topics) * len(PACKAGES) * len(models)
    alpha = 0.05 / n_tests
    width = max(len(MODEL_LABELS[m]) for m in models)
    n_columns = 1 + 3 * len(PACKAGES)

    body = []
    for i, topic in enumerate(topics):
        if i:
            body.append(r"\addlinespace")
        body.append(r"\multicolumn{" + str(n_columns) + r"}{l}{\textit{"
                    + TOPIC_LABELS[topic] + r"}} \\")
        body += [_row(name, width,
                      [x for p in PACKAGES
                       for x in triple(results[topic][p][name], alpha)])
                 for name in models]
    return table(caption("message package (three packages per topic)",
                         "package", payload["n_resamples"], n_tests),
                 "tab:si_permutation_package",
                 [f"Package {p[-1]}" for p in PACKAGES], body)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild", action="store_true",
                        help=f"render the tables from {JSON_PATH.name} instead of "
                             "recomputing the permutation tests")
    parser.add_argument("--json", default=JSON_PATH, type=str)
    args = parser.parse_args()

    ensure_output_dirs()
    if args.rebuild:
        try:
            with open(args.json, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            raise SystemExit(f"no results at {args.json}; run without --rebuild first")
        if "package" not in payload:
            raise SystemExit(f"{args.json} predates the package table; "
                             "run without --rebuild to recompute it")
    else:
        payload = compute()
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.json}")

    for path, text in ((TOPIC_TEX_PATH, render_topic(payload)),
                       (PACKAGE_TEX_PATH, render_package(payload))):
        print(text)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
