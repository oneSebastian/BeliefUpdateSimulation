"""Combined effect-size and test-statistic report, assembled from source.

One table per analysis: the test statistic (with the p-value), the effect size,
and -- in the final section -- the sensitivity analysis (minimum detectable
effect at 80% power) next to the observed effects. Each number is read from the
module that owns the test rather than recomputed here:

    permutation    scripts.stats.analyze via belief_update_sim.permutation_test
    chi-squared    scripts.stats.post_stance_distribution   (Cramer's V)
    Brown-Forsythe scripts.stats.belief_change_variability  (variance ratio)
    Brown-Forsythe scripts.stats.comment_rank_variability   (variance ratio)
    Kendall tau    scripts.stats.comment_rank_correlation   (mean tau)
    sensitivity    scripts.stats.power_sensitivity          (minimum detectable)

Both Brown-Forsythe tests are two-sided.

    python -m scripts.stats.effect_size_report
"""

import argparse
import contextlib
import io

import pandas as pd

from belief_update_sim.comment_ranks import HUMAN, MODELS
from belief_update_sim.config import RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data
from belief_update_sim.permutation_test import cohens_dz, fast_permutation_test

from scripts.stats import belief_change_variability as bcv
from scripts.stats import comment_rank_correlation as crc
from scripts.stats import comment_rank_variability as crv
from scripts.stats import post_stance_distribution as psd
from scripts.stats import power_sensitivity as ps

ALPHA = 0.05 / 6


def permutation_rows():
    """Per-model paired stance difference + paired Cohen's d_z + permutation p."""
    human = load_human_normalized_data()
    rows = {}
    for name, filename in MODELS.items():
        model = load_normalized_data(RESULTS_DIR / filename)
        merged = pd.merge(human, model, on=["persona_id", "topic"],
                          suffixes=("_h", "_m"), validate="one_to_one")
        diff = (merged["new_belief_h"] - merged["new_belief_m"]).dropna()
        s1 = merged.loc[diff.index, "new_belief_h"]
        s2 = merged.loc[diff.index, "new_belief_m"]
        obs, p, _ = fast_permutation_test(s1, s2)
        rows[name] = {"n": int(diff.size), "mean_diff": float(diff.mean()),
                      "sd_diff": float(diff.std(ddof=1)),
                      "d_z": cohens_dz(s1, s2), "p": float(p)}
    return rows


def gather():
    perm = permutation_rows()
    _, _, chi = psd.compute()
    sd_human_delta, n_human_delta, bf_delta = bcv.compute()

    means, comments = crv.collect()
    with contextlib.redirect_stdout(io.StringIO()):        # render() also prints
        _, bf_rank = crv.render(means, comments)

    per_model_tau, _, _, _ = crc.compute()
    tau = {name: crc.summarize(taus) for name, taus in per_model_tau.items()}

    sens = ps.compute()
    return {"perm": perm, "chi": chi, "sd_human_delta": sd_human_delta,
            "n_human_delta": n_human_delta, "bf_delta": bf_delta,
            "bf_rank": bf_rank, "tau": tau, "sens": sens}


def render(g):
    L = []

    def out(text=""):
        L.append(text)

    perm, chi = g["perm"], g["chi"]
    bf_delta, bf_rank, tau = g["bf_delta"], g["bf_rank"], g["tau"]
    sd_human_delta, n_human_delta = g["sd_human_delta"], g["n_human_delta"]
    sens = g["sens"]

    out("=" * 78)
    out("BELIEF UPDATE SIMULATION -- EFFECT SIZES, TEST STATISTICS, SENSITIVITY")
    out("=" * 78)
    out()
    out("Design: 391 participants x 3 topics = 1173 observations.")
    out(f"Bonferroni threshold alpha = 0.05/6 = {ALPHA:.4f}; power target = 0.80.")
    out()

    # 1. permutation --------------------------------------------------------
    out("-" * 78)
    out("1. PERMUTATION TEST -- paired post-stance difference (human - LLM)")
    out("   statistic: mean of paired differences; two-sided sign-flip "
        "permutation (1,000,000 resamples)")
    out("   effect size: paired Cohen's d_z = mean(diff) / sd(diff)")
    out("-" * 78)
    out(f"{'model':<26}{'n':>6}{'mean diff':>11}{'sd diff':>10}{'d_z':>9}{'p':>12}")
    for name in MODELS:
        r = perm[name]
        out(f"{name:<26}{r['n']:>6}{r['mean_diff']:>+11.4f}{r['sd_diff']:>10.4f}"
            f"{r['d_z']:>+9.4f}{r['p']:>12.4g}")
    out()

    # 2. chi-squared --------------------------------------------------------
    out("-" * 78)
    out("2. CHI-SQUARED TEST OF INDEPENDENCE -- post-stance distribution")
    out("   statistic: chi-squared on 2 x 5 contingency table, df = 4; "
        "non-directional")
    out("   effect size: Cramer's V = sqrt(chi2 / N)")
    out("-" * 78)
    out(f"{'model':<26}{'chi2':>10}{'df':>4}{'p':>12}{'CramerV':>10}"
        f"{'min E':>9}{'N':>7}")
    for name in MODELS:
        r = chi[name]
        n_total = r["n_human"] + r["n_model"]
        out(f"{name:<26}{r['chi2']:>10.3f}{r['dof']:>4}{r['p_value']:>12.4g}"
            f"{r['cramers_v']:>10.4f}{r['min_expected']:>9.1f}{n_total:>7}")
    out()

    # 3. brown-forsythe, belief change -------------------------------------
    out("-" * 78)
    out("3. BROWN-FORSYTHE TEST -- spread of |belief change| = |post - initial|")
    out("   statistic: median-centred Levene W ~ F(1, N-2); two-sided")
    out("   effect size: variance ratio and SD ratio, human / model")
    out(f"   human: n = {n_human_delta}, SD = {sd_human_delta:.4f}")
    out("-" * 78)
    out(f"{'model':<26}{'sd model':>10}{'var ratio':>11}{'sd ratio':>10}"
        f"{'W':>9}{'df2':>7}{'p':>12}")
    for name in MODELS:
        r = bf_delta[name]
        df2 = n_human_delta + r["n_model"] - 2
        out(f"{name:<26}{r['sd_model']:>10.4f}{r['variance_ratio']:>11.3f}"
            f"{r['sd_ratio']:>10.3f}{r['W']:>9.3f}{df2:>7}{r['p_value']:>12.4g}")
    out()

    # 4. brown-forsythe, mean ranks ----------------------------------------
    out("-" * 78)
    out("4. BROWN-FORSYTHE TEST -- spread of the 27 comment mean-ranks")
    out("   statistic: median-centred Levene W ~ F(1, 52); two-sided")
    out("   effect size: variance ratio and SD ratio, model / human")
    out(f"   human: n = {bf_rank['n_comments']} comments, "
        f"SD = {bf_rank['sd_human']:.4f}")
    out("-" * 78)
    out(f"{'model':<26}{'sd model':>10}{'var ratio':>11}{'sd ratio':>10}"
        f"{'W':>9}{'p':>12}")
    for name in MODELS:
        r = bf_rank["models"][name]
        out(f"{name:<26}{r['sd']:>10.4f}{r['variance_ratio']:>11.3f}"
            f"{r['sd_ratio']:>10.3f}{r['W']:>9.3f}{r['p_value']:>12.4g}")
    out()

    # 5. kendall tau --------------------------------------------------------
    out("-" * 78)
    out("5. KENDALL RANK CORRELATION -- human vs LLM comment ranking")
    out("   tau = (Nc - Nd)/N per human-LLM pair; tau in {-1, -1/3, 1/3, 1}")
    out("   effect size: mean tau IS the effect size (a rank correlation, like")
    out("   Pearson r); reported descriptively -- no significance test / p-value")
    out("-" * 78)
    out(f"{'model':<26}{'n pairs':>9}{'mean tau':>11}{'sd':>9}{'se':>9}"
        f"{'95% CI':>22}")
    for name in MODELS:
        s = tau[name]
        ci = f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]"
        out(f"{name:<26}{s['n']:>9}{s['mean']:>+11.5f}{s['sd']:>9.4f}"
            f"{s['se']:>9.4f}{ci:>22}")
    out()

    # 6. sensitivity vs observed -------------------------------------------
    out("-" * 78)
    out("6. SENSITIVITY -- minimum detectable effect (MDE) at 80% power, "
        f"alpha = {ALPHA:.4f}")
    out("   nominal n = observation level; independent n = 391 participants")
    out("-" * 78)
    p = sens["permutation_dz"]
    c = sens["chi2_cramers_v"]
    b = sens["brown_forsythe_belief_change_sd_ratio"]
    br = sens["brown_forsythe_mean_rank_sd_ratio"]["fixed_27"]
    k = sens["kendall_tau"]

    perm_dz = [abs(perm[n]["d_z"]) for n in MODELS]
    chi_v = [chi[n]["cramers_v"] for n in MODELS]
    bf_sd = [bf_delta[n]["sd_ratio"] for n in MODELS]
    rank_sd = [bf_rank["models"][n]["sd_ratio"] for n in MODELS]
    tau_abs = [abs(tau[n]["mean"]) for n in MODELS]

    def srow(label, nom, ind, obs, dec):
        fn = "" if nom is None else f"{nom:.{dec}f}"
        out(f"{label:<40}{fn:>11}{ind:>11.{dec}f}"
            f"{min(obs):>11.{dec}f}{max(obs):>11.{dec}f}")

    out(f"{'metric':<40}{'MDE nom':>11}{'MDE indep':>11}"
        f"{'obs min':>11}{'obs max':>11}")
    srow("permutation  detectable d_z", p["nominal"], p["independent"],
         perm_dz, 4)
    srow("chi-squared  detectable Cramer V", c["nominal"], c["independent"],
         chi_v, 4)
    srow("BF |belief change|  SD ratio", b["nominal"], b["independent"],
         bf_sd, 3)
    srow("BF |belief change|  variance ratio", b["nominal"] ** 2,
         b["independent"] ** 2, [x ** 2 for x in bf_sd], 3)
    srow("BF mean-rank spread  SD ratio (n=27)", None, br, rank_sd, 3)
    srow("Kendall  detectable mean |tau|", k["nominal"], k["independent"],
         tau_abs, 4)
    out()
    out("MDE = minimum detectable effect at 80% power; obs min/max = smallest and")
    out("largest observed effect across the six models (|d_z| and |mean tau| as")
    out("absolute values). An observed effect below the MDE was not resolvable at")
    out("that sample size. The Kendall row is the MDE for a HYPOTHETICAL one-sample")
    out("test of mean tau vs 0 -- that test is not run in the paper and is shown")
    out("only for the power reporting; tau itself remains the reported effect size.")
    out()

    return "\n".join(L) + "\n"


def main():
    argparse.ArgumentParser().parse_args()
    text = render(gather())
    print(text)

    ensure_output_dirs()
    (STATS_OUTPUT_DIR / "effect_size_report.txt").write_text(text, encoding="utf-8")
    print(f"wrote {STATS_OUTPUT_DIR / 'effect_size_report.txt'}")


if __name__ == "__main__":
    main()
