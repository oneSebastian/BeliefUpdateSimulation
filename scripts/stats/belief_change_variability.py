"""Is human belief change more variable than the LLMs' belief change?

Absolute belief change is |post-stance - initial stance| for each participant-
topic pair. Its spread is compared between humans and each model with a
Brown-Forsythe test (Levene centred on the median, robust to the non-normal,
bounded distribution of |delta|):

    H0: var(human |delta|) == var(model |delta|)
    H1: var(human) >  var(model)      -- humans change their belief more variably

The alternative is directional, so this is a **one-sided** test. scipy's Levene
returns a two-sided p; the one-sided p is half of it once the observed direction
matches (sd(human) > sd(model)), and its complement otherwise so a result in the
wrong direction can never read as significant.

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


def human_abs_delta():
    df = pd.read_csv(MERGED_HUMAN_CSV)
    delta = (df["final_belief_normalized"] - df["initial_belief_normalized"]).abs()
    return delta.dropna().to_numpy()


def model_abs_delta(path):
    df = pd.read_excel(path)
    # |post - init| is the same whether or not the pro/con flip is applied, since
    # both terms flip together; use the raw recorded stances directly
    delta = (df["llm_new_belief"] - df["init_belief"]).abs()
    return delta.dropna().to_numpy()


def compute():
    human = human_abs_delta()
    sd_human = float(pd.Series(human).std())          # ddof=1, matches the figure

    results = {}
    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        model = model_abs_delta(path)
        sd_model = float(pd.Series(model).std())
        w, p_two = levene(human, model, center="median")
        larger = sd_human > sd_model
        p_one = p_two / 2 if larger else 1 - p_two / 2
        results[name] = {
            "n_model": int(model.size), "sd_model": sd_model,
            "W": float(w), "p_two_sided": float(p_two), "p_one_sided": float(p_one),
            "human_more_variable": bool(larger),
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
    out("H0: var(human) == var(model)     H1: var(human) > var(model)  (one-sided)")
    out("=" * 96)
    out()
    out(f"human: n = {n_human}, SD = {sd_human:.4f}")
    out()
    out(f"{'model':<26}{'n':>7}{'SD(model)':>11}{'W':>10}"
        f"{'p 2-sided':>14}{'p 1-sided':>14}   H1")
    out("-" * 96)
    for name, r in results.items():
        out(f"{name:<26}{r['n_model']:>7}{r['sd_model']:>11.4f}{r['W']:>10.3f}"
            f"{r['p_two_sided']:>14.3e}{r['p_one_sided']:>14.3e}"
            f"   {'yes' if r['p_one_sided'] < alpha else 'no'}")
    out("-" * 96)
    out(f"one-sided test; Bonferroni across {len(results)} models: alpha = {alpha:.4f}")

    worst = max(r["p_one_sided"] for r in results.values())
    out()
    out(f"largest one-sided p across all models: {worst:.3e}")
    if all(r["p_one_sided"] < alpha for r in results.values()):
        out("=> human belief change is significantly more variable than every model")
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
