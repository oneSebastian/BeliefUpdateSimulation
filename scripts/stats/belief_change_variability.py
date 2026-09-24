"""Is human belief change more variable than the LLMs' belief change?

Absolute belief change is |post-stance - initial stance| for each participant-
topic pair. Its spread is compared between humans and each model with a
Brown-Forsythe test (Levene centred on the median, robust to the non-normal,
bounded distribution of |delta|):

    H0: var(human |delta|) == var(model |delta|)
    H1: var(human) != var(model)      -- the spreads differ

The alternative is non-directional, so this is a **two-sided** test: scipy's
Levene (median-centred) returns the two-sided p directly. The observed direction
is reported descriptively as the SD/variance ratio (human / model) and the
`human_more_variable` flag.

|delta| is invariant to the pro/con sign flip (both terms flip together), so the
absolute change is computed from the raw recorded stances and equals the value
the belief-change figure reports.

    python -m scripts.stats.belief_change_variability
"""

import argparse
import json

import pandas as pd
from scipy.stats import levene

from belief_update_sim.comment_ranks import MODELS
from belief_update_sim.config import MERGED_HUMAN_CSV, RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.grouping import ALL


def human_abs_delta(group=ALL):
    df = group.filter(pd.read_csv(MERGED_HUMAN_CSV))
    delta = (df["final_belief_normalized"] - df["initial_belief_normalized"]).abs()
    return delta.dropna().to_numpy()


def model_abs_delta(path, group=ALL):
    df = group.filter(pd.read_excel(path))
    # |post - init| is the same whether or not the pro/con flip is applied, since
    # both terms flip together; use the raw recorded stances directly
    delta = (df["llm_new_belief"] - df["init_belief"]).abs()
    return delta.dropna().to_numpy()


def compute(group=ALL):
    human = human_abs_delta(group)
    sd_human = float(pd.Series(human).std())          # ddof=1, matches the figure

    results = {}
    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        model = model_abs_delta(path, group)
        sd_model = float(pd.Series(model).std())
        w, p_two = levene(human, model, center="median")
        sd_ratio = sd_human / sd_model
        results[name] = {
            "n_model": int(model.size), "sd_model": sd_model,
            "sd_ratio": sd_ratio, "variance_ratio": sd_ratio ** 2,
            "W": float(w), "p_value": float(p_two),
            "human_more_variable": bool(sd_human > sd_model),
        }
    return sd_human, int(human.size), results


def render(sd_human, n_human, results):
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    alpha = 0.05 / len(results)
    out("=" * 96)
    out("Brown-Forsythe test on absolute belief change |post - initial|")
    out("H0: var(human) == var(model)     H1: var(human) != var(model)  (two-sided)")
    out("=" * 96)
    out()
    out(f"human: n = {n_human}, SD = {sd_human:.4f}")
    out()
    out(f"{'model':<26}{'n':>7}{'SD(model)':>11}{'var ratio':>11}{'W':>10}"
        f"{'p':>14}   sig")
    out("-" * 96)
    for name, r in results.items():
        out(f"{name:<26}{r['n_model']:>7}{r['sd_model']:>11.4f}"
            f"{r['variance_ratio']:>11.3f}{r['W']:>10.3f}{r['p_value']:>14.3e}"
            f"   {'yes' if r['p_value'] < alpha else 'no'}")
    out("-" * 96)
    out(f"two-sided test; Bonferroni across {len(results)} models: alpha = {alpha:.4f}")
    out("var ratio is human/model; human_more_variable holds for every model")

    worst = max(r["p_value"] for r in results.values())
    out()
    out(f"largest p across all models: {worst:.3e}")
    if all(r["p_value"] < alpha for r in results.values()):
        out("=> human and model belief-change spreads differ significantly for every model")
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser().parse_args()
    sd_human, n_human, results = compute()
    text = render(sd_human, n_human, results)

    ensure_output_dirs()
    (STATS_OUTPUT_DIR / "belief_change_variability.txt").write_text(text, encoding="utf-8")
    payload = {"sd_human": sd_human, "n_human": n_human, "models": results}
    (STATS_OUTPUT_DIR / "belief_change_variability.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {STATS_OUTPUT_DIR / 'belief_change_variability.txt'}")
    print(f"wrote {STATS_OUTPUT_DIR / 'belief_change_variability.json'}")


if __name__ == "__main__":
    main()
