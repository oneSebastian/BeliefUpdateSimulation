"""Do LLM post-stance distributions differ from the human distribution?

For each model, the post-stance (on the five-point Likert scale, in the
normalised pro-topic frame) is tabulated for humans and for the model, and the
two distributions are compared with a chi-squared test of independence on the
2 x 5 contingency table.

The test is **two-sided** in the sense that its alternative is
non-directional ("the distributions differ"), with no greater/less direction to
take a side on.

    python -m scripts.stats.post_stance_distribution
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from belief_update_sim.comment_ranks import MODELS
from belief_update_sim.config import RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data

LIKERT = [-2, -1, 0, 1, 2]
LABELS = ["SD", "D", "N", "A", "SA"]


def post_stance(series):
    return pd.to_numeric(series, errors="coerce").dropna().astype(int)


def counts(series):
    return [int((series == v).sum()) for v in LIKERT]


def compute():
    human = post_stance(load_human_normalized_data()["new_belief"])
    human_counts = counts(human)

    results = {}
    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        model = post_stance(load_normalized_data(path)["new_belief"])
        model_counts = counts(model)

        table = np.array([human_counts, model_counts])
        chi2, p, dof, expected = chi2_contingency(table)
        n_total = int(table.sum())
        # 2 rows -> min(r, c) - 1 = 1, so Cramer's V = sqrt(chi2 / N)
        cramers_v = float((chi2 / n_total) ** 0.5)
        results[name] = {
            "n_human": int(human.size), "n_model": int(model.size),
            "human_counts": human_counts, "model_counts": model_counts,
            "chi2": float(chi2), "dof": int(dof), "p_value": float(p),
            "cramers_v": cramers_v, "min_expected": float(expected.min()),
        }
    return human_counts, int(human.size), results


def render(human_counts, n_human, results):
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 96)
    out("Chi-squared test of independence: human vs LLM post-stance distribution")
    out("H0: the post-stance distribution is the same for humans and the model")
    out("H1: the distributions differ (non-directional -> two-sided)")
    out("post-stance on the normalised 5-point scale; 2 x 5 contingency table, dof = 4")
    out("=" * 96)
    out()

    header = "".join(f"{l:>7}" for l in LABELS)
    out(f"{'':<26}{header}{'total':>9}")
    out(f"{'Human':<26}" + "".join(f"{c:>7}" for c in human_counts) + f"{n_human:>9}")
    out("-" * 96)

    for name, r in results.items():
        out(f"{name:<26}" + "".join(f"{c:>7}" for c in r['model_counts'])
            + f"{r['n_model']:>9}")
    out()

    out(f"{'model':<26}{'chi2':>12}{'dof':>6}{'p':>14}{'CramerV':>10}{'min E':>10}")
    out("-" * 96)
    for name, r in results.items():
        out(f"{name:<26}{r['chi2']:>12.3f}{r['dof']:>6}{r['p_value']:>14.3e}"
            f"{r['cramers_v']:>10.4f}{r['min_expected']:>10.1f}")
    out("-" * 96)
    out("test is two-sided (non-directional); min E is the smallest expected cell "
        "count\n(the chi-squared approximation is reliable when this is >= 5)")

    worst = max(r["p_value"] for r in results.values())
    out()
    out(f"largest p across all models: {worst:.3e}")
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser().parse_args()
    human_counts, n_human, results = compute()
    text = render(human_counts, n_human, results)

    ensure_output_dirs()
    (STATS_OUTPUT_DIR / "post_stance_distribution.txt").write_text(text, encoding="utf-8")
    (STATS_OUTPUT_DIR / "post_stance_distribution.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {STATS_OUTPUT_DIR / 'post_stance_distribution.txt'}")
    print(f"wrote {STATS_OUTPUT_DIR / 'post_stance_distribution.json'}")


if __name__ == "__main__":
    main()
