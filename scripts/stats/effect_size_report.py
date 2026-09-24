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

    python -m scripts.stats.effect_size_report                      # all 1173 obs
    python -m scripts.stats.effect_size_report --group-by topic     # 3 subsets
    python -m scripts.stats.effect_size_report --group-by topic-package   # 9

``--group-by`` reruns every one of the six analyses inside each cell of the
design (see :mod:`belief_update_sim.grouping`) and writes the results under
``outputs/stats/grouped/``. The bare invocation is untouched by it: it takes
the same code path as before and still overwrites the published
``outputs/stats/effect_size_report.txt``.

``--output-dir`` sends any grouping, including ``overall``, to
``DIR/<grouping>/`` instead. That is how the three runs are kept together for
comparison without the whole-design one overwriting the published report:

    python -m scripts.stats.effect_size_report --group-by overall \\
        --output-dir outputs/stats/grouped
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import pandas as pd

from belief_update_sim.comment_ranks import HUMAN, MODELS
from belief_update_sim.config import (GROUPED_STATS_DIR, RESULTS_DIR, STATS_OUTPUT_DIR,
                                      ensure_output_dirs)
from belief_update_sim.data_loading import load_human_normalized_data, load_normalized_data
from belief_update_sim.grouping import ALL, GROUPINGS, Group, design_counts
from belief_update_sim.permutation_test import cohens_dz, fast_permutation_test

from scripts.stats import belief_change_variability as bcv
from scripts.stats import comment_rank_correlation as crc
from scripts.stats import comment_rank_variability as crv
from scripts.stats import post_stance_distribution as psd
from scripts.stats import power_sensitivity as ps

ALPHA = 0.05 / 6

MIN_EXPECTED = 5        # below this the chi-squared approximation is unreliable
N_COMMENTS_FULL = 27    # comment mean-ranks over the whole design


def permutation_rows(group=ALL):
    """Per-model paired stance difference + paired Cohen's d_z + permutation p.

    Only the human side is subset: the model frame is merged whole, so a
    participant-topic whose package disagrees between the two sources would
    show up as a package mismatch rather than being silently dropped by the
    inner merge before anything could notice.
    """
    human = group.filter(load_human_normalized_data())
    rows = {}
    for name, filename in MODELS.items():
        model = load_normalized_data(RESULTS_DIR / filename)
        merged = pd.merge(human, model, on=["persona_id", "topic"],
                          suffixes=("_h", "_m"), validate="one_to_one")
        mismatched = merged["package_h"] != merged["package_m"]
        if mismatched.any():
            raise ValueError(
                f"{name}: {int(mismatched.sum())} participant-topics were shown "
                "a different comment package by the human study and the model "
                "run; a package subset would not be comparing the same stimuli."
            )
        diff = (merged["new_belief_h"] - merged["new_belief_m"]).dropna()
        s1 = merged.loc[diff.index, "new_belief_h"]
        s2 = merged.loc[diff.index, "new_belief_m"]
        obs, p, _ = fast_permutation_test(s1, s2)
        rows[name] = {"n": int(diff.size), "mean_diff": float(diff.mean()),
                      "sd_diff": float(diff.std(ddof=1)),
                      "d_z": cohens_dz(s1, s2), "p": float(p)}
    return rows


def gather(group=ALL, n_groups=1):
    perm = permutation_rows(group)
    _, _, chi = psd.compute(group)
    sd_human_delta, n_human_delta, bf_delta = bcv.compute(group)

    means, comments = crv.collect(group)
    with contextlib.redirect_stdout(io.StringIO()):        # render() also prints
        _, bf_rank = crv.render(means, comments)

    per_model_tau, _, _, _ = crc.compute(group)
    tau = {name: crc.summarize(taus) for name, taus in per_model_tau.items()}

    sens = ps.compute(group)
    return {"perm": perm, "chi": chi, "sd_human_delta": sd_human_delta,
            "n_human_delta": n_human_delta, "bf_delta": bf_delta,
            "bf_rank": bf_rank, "tau": tau, "sens": sens,
            "group": group, "n_groups": n_groups,
            "counts": design_counts(group)}


def render(g):
    L = []

    def out(text=""):
        L.append(text)

    perm, chi = g["perm"], g["chi"]
    bf_delta, bf_rank, tau = g["bf_delta"], g["bf_rank"], g["tau"]
    sd_human_delta, n_human_delta = g["sd_human_delta"], g["n_human_delta"]
    sens = g["sens"]
    group = g.get("group", ALL)
    n_groups = g.get("n_groups", 1)

    out("=" * 78)
    out("BELIEF UPDATE SIMULATION -- EFFECT SIZES, TEST STATISTICS, SENSITIVITY")
    out("=" * 78)
    out()
    if group.is_all:
        out("Design: 391 participants x 3 topics = 1173 observations.")
    else:
        counts = g["counts"]
        out(f"SUBSET: {group.title}")
        out(f"Design: {counts['n_participants']} participants x 1 cell = "
            f"{counts['n_obs']} observations. Each participant")
        out("appears once in this cell, so the nominal and independent sample "
            "sizes")
        out("coincide and no within-participant clustering is left to correct for.")
    out(f"Bonferroni threshold alpha = 0.05/6 = {ALPHA:.4f}; power target = 0.80.")
    if n_groups > 1:
        family = 6 * n_groups
        out("That alpha is across the 6 models, as in the whole-design report.")
        out(f"Treating all {n_groups} subsets as one family instead gives "
            f"alpha = 0.05/{family} = {0.05 / family:.5f};")
        out("both are marked in the cross-subset overview, so the choice stays yours.")
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
    out(f"   nominal n = observation level ({sens.get('n_obs', 1173)}); "
        f"independent n = {sens.get('n_participants', 391)} participants")
    out("-" * 78)
    p = sens["permutation_dz"]
    c = sens["chi2_cramers_v"]
    b = sens["brown_forsythe_belief_change_sd_ratio"]
    rank_mde = sens["brown_forsythe_mean_rank_sd_ratio"]
    br, n_rank = rank_mde["fixed_n"], rank_mde["n"]
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
    srow(f"BF mean-rank spread  SD ratio (n={n_rank})", None, br, rank_sd, 3)
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

    if not group.is_all:
        out("-" * 78)
        out(f"READING THIS SUBSET -- {group.title}")
        out("-" * 78)
        for line in subset_caveats(g):
            out(line)
        out()

    return "\n".join(L) + "\n"


def subset_caveats(g):
    """What a reader has to know before taking a subset table at face value.

    These are consequences of the smaller n, not of anything going wrong, so
    they are stated on every subset report rather than raised as warnings.
    """
    chi, bf_rank, counts = g["chi"], g["bf_rank"], g["counts"]
    sens = g["sens"]
    lines = [
        f"n = {counts['n_obs']} here against 1173 for the whole design, so every",
        "test is correspondingly less sensitive. Section 6 re-solves the minimum",
        "detectable effect at this n -- compare each observed effect against it",
        "before reading a non-significant result as an absence of a difference.",
        "",
        f"Paired d_z below {sens['permutation_dz']['nominal']:.4f} is not "
        "resolvable at this n.",
    ]

    worst_e = min(r["min_expected"] for r in chi.values())
    if worst_e < MIN_EXPECTED:
        lines += [
            "",
            f"Section 2: the smallest expected cell count falls to {worst_e:.1f} "
            f"(< {MIN_EXPECTED}) for at",
            "least one model, so the chi-squared approximation is no longer "
            "reliable",
            "here. Treat those p-values as indicative; reporting them would need "
            "an",
            "exact or Monte-Carlo test instead.",
        ]

    n_comments = bf_rank["n_comments"]
    if n_comments < N_COMMENTS_FULL:
        lines += [
            "",
            f"Section 4 compares the spread of only {n_comments} comment "
            f"mean-ranks per group",
            f"(the {N_COMMENTS_FULL} comments are 3 topics x 3 packages x 3 "
            "messages, so a subset holds",
            "a third or a ninth of them). A Brown-Forsythe test on groups that "
            "small has",
            "almost no power; the SD and variance ratios remain descriptive, the "
            "p-values",
            "essentially uninformative.",
        ]

    return lines


def summary(group, g):
    """The whole payload for one subset, as JSON-serialisable data."""
    return {
        "group": group.label, "title": group.title,
        "topic": group.topic, "package": group.package,
        "counts": g["counts"],
        "permutation": g["perm"],
        "chi_squared": g["chi"],
        "brown_forsythe_belief_change": {
            "sd_human": g["sd_human_delta"], "n_human": g["n_human_delta"],
            "models": g["bf_delta"],
        },
        "brown_forsythe_mean_ranks": g["bf_rank"],
        "kendall_tau": g["tau"],
        "sensitivity": g["sens"],
    }


def overview(results):
    """One matrix per statistic: subsets down the side, models across the top.

    The per-subset reports hold everything; this is the view that makes a
    difference *between* subsets visible, which is the reason for running them
    separately in the first place.
    """
    L = []

    def out(text=""):
        L.append(text)

    n_groups = len(results)
    family_alpha = 0.05 / (6 * n_groups)
    width = max(len(group.label) for group, _ in results) + 2

    def marks(p):
        if p < family_alpha:
            return "**"
        if p < ALPHA:
            return "*"
        return ""

    def p_cell(p):
        return f"{p:.3g}{marks(p)}"

    def to_4dp(value, signed=False):
        """Four decimal places, or '<0.0001' when the value rounds away.

        A permutation p of exactly 0 means no resample out of a million
        reached the observed statistic, not that the probability is zero, so
        printing '0' would overstate it.
        """
        if round(value, 4) == 0:
            return "<0.0001"
        return f"{value:+.4f}" if signed else f"{value:.4f}"

    def p_cell_4dp(p):
        return to_4dp(p) + marks(p)

    def table(title, note, cell):
        out("-" * (width + 13 * len(MODELS)))
        out(title)
        if note:
            out(f"   {note}")
        out("-" * (width + 13 * len(MODELS)))
        out(f"{'subset':<{width}}" + "".join(f"{n[:12]:>13}" for n in MODELS))
        for group, g in results:
            out(f"{group.label:<{width}}"
                + "".join(f"{cell(g, name):>13}" for name in MODELS))
        out()

    out("=" * (width + 13 * len(MODELS)))
    out("CROSS-SUBSET OVERVIEW")
    out("=" * (width + 13 * len(MODELS)))
    out()
    out(f"{'subset':<{width}}{'n obs':>9}{'n participants':>16}{'n comments':>13}")
    for group, g in results:
        out(f"{group.label:<{width}}{g['counts']['n_obs']:>9}"
            f"{g['counts']['n_participants']:>16}"
            f"{g['bf_rank']['n_comments']:>13}")
    out()
    out(f"*  p < {ALPHA:.5f}   (0.05/6, across the six models within a subset)")
    out(f"** p < {family_alpha:.5f}   (0.05/{6 * n_groups}, "
        f"across all {n_groups} subsets as one family)")
    out("In section 1, '<0.0001' means the value rounds to 0.0000 at four "
        "decimal places.")
    out()

    table("1. PERMUTATION -- mean paired final-stance difference (human - LLM)",
          "raw scale points, normalised pro-topic 5-point scale; the statistic "
          "the test permutes",
          lambda g, n: to_4dp(g['perm'][n]['mean_diff'], signed=True))
    table("1. PERMUTATION -- paired Cohen's d_z (human - LLM post-stance)",
          "the same difference divided by its SD across pairs",
          lambda g, n: f"{g['perm'][n]['d_z']:+.4f}")
    table("1. PERMUTATION -- p",
          "1,000,000 sign-flip resamples",
          lambda g, n: p_cell_4dp(g['perm'][n]['p']))
    table("2. CHI-SQUARED -- Cramer's V (post-stance distribution)", None,
          lambda g, n: f"{g['chi'][n]['cramers_v']:.4f}")
    table("2. CHI-SQUARED -- p", None,
          lambda g, n: p_cell(g['chi'][n]['p_value']))
    table("2. CHI-SQUARED -- smallest expected cell count",
          f"the approximation needs >= {MIN_EXPECTED}",
          lambda g, n: f"{g['chi'][n]['min_expected']:.1f}")
    table("3. BROWN-FORSYTHE |belief change| -- variance ratio (human/model)", None,
          lambda g, n: f"{g['bf_delta'][n]['variance_ratio']:.3f}")
    table("3. BROWN-FORSYTHE |belief change| -- p", None,
          lambda g, n: p_cell(g['bf_delta'][n]['p_value']))
    table("4. BROWN-FORSYTHE mean ranks -- variance ratio (model/human)",
          "on very few comments per subset; see each report's caveats",
          lambda g, n: f"{g['bf_rank']['models'][n]['variance_ratio']:.3f}")
    table("4. BROWN-FORSYTHE mean ranks -- p",
          "on very few comments per subset; see each report's caveats",
          lambda g, n: p_cell(g['bf_rank']['models'][n]['p_value']))
    table("5. KENDALL -- mean tau (descriptive; no p-value)", None,
          lambda g, n: f"{g['tau'][n]['mean']:+.4f}")
    table("6. SENSITIVITY -- MDE for d_z at this subset's n",
          "identical across models; an observed |d_z| below it is unresolvable",
          lambda g, n: f"{g['sens']['permutation_dz']['nominal']:.4f}")

    return "\n".join(L) + "\n"


#: --group-by value -> the subdirectory its run writes into
SUBDIR = {"overall": "whole_design",
          "topic": "by_topic",
          "topic-package": "by_topic_package"}


def payload_from_summary(blob):
    """Invert `summary()`: a saved all_subsets.json back into render payloads.

    Everything `render` and `overview` read is in that file, so the text
    reports can be rewritten after a formatting change without paying for the
    permutations and the power simulation a second time.
    """
    results = []
    for record in blob.values():
        group = Group(topic=record["topic"], package=record["package"])
        bf = record["brown_forsythe_belief_change"]
        results.append((group, {
            "perm": record["permutation"],
            "chi": record["chi_squared"],
            "sd_human_delta": bf["sd_human"],
            "n_human_delta": bf["n_human"],
            "bf_delta": bf["models"],
            "bf_rank": record["brown_forsythe_mean_ranks"],
            "tau": record["kendall_tau"],
            "sens": record["sensitivity"],
            "group": group, "n_groups": len(blob),
            "counts": record["counts"],
        }))
    return results


def write_subset_report(group, g, out_dir):
    """Write one subset's report and return its text.

    Called as soon as a subset finishes rather than at the end of the run: the
    nine-cell grouping takes some twenty minutes, and a failure in the last
    cell should not throw away the eight reports already computed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    text = render(g)
    path = out_dir / f"{group.label}.txt"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return text


def write_aggregate_reports(results, sections, out_dir, write_json=True):
    """Write the cross-subset overview and the machine-readable payload."""
    written = []
    if len(results) > 1:
        # with a single subset these two would just duplicate its own report
        overview_text = overview(results)
        (out_dir / "overview.txt").write_text(overview_text, encoding="utf-8")
        (out_dir / "all_subsets.txt").write_text(
            overview_text + "\n" + "\n".join(sections), encoding="utf-8")
        written += ["overview.txt", "all_subsets.txt"]
        print(overview_text)

    if write_json:
        (out_dir / "all_subsets.json").write_text(
            json.dumps({group.label: summary(group, g) for group, g in results},
                       indent=2, default=float), encoding="utf-8")
        written.append("all_subsets.json")

    for filename in written:
        print(f"wrote {out_dir / filename}")
    return written


def run_grouping(name, base_dir):
    """Run every analysis in each subset of `name` and write `base_dir/<sub>/`.

    Returns the directory written. One grouping per directory, so the three
    runs sit side by side and none of them overwrites another.
    """
    groups = GROUPINGS[name]
    out_dir = base_dir / SUBDIR[name]

    results, sections = [], []
    for i, group in enumerate(groups, start=1):
        # each subset re-runs six analyses -- a million-resample permutation
        # per model, and a simulated power solve -- so say where we are
        print(f"[{i}/{len(groups)}] computing {group.title} ...", flush=True)
        g = gather(group, n_groups=len(groups))
        results.append((group, g))
        sections.append(write_subset_report(group, g, out_dir))

    write_aggregate_reports(results, sections, out_dir)
    return out_dir


def rebuild_grouping(name, base_dir):
    """Rewrite a finished run's text reports from its all_subsets.json."""
    out_dir = base_dir / SUBDIR[name]
    path = out_dir / "all_subsets.json"
    if not path.exists():
        raise SystemExit(
            f"nothing to rebuild: {path} does not exist. It is written at the "
            f"end of a run, so run --group-by {name} without --rebuild first."
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    results = payload_from_summary(blob)
    sections = [write_subset_report(group, g, out_dir) for group, g in results]
    write_aggregate_reports(results, sections, out_dir, write_json=False)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--group-by", choices=list(GROUPINGS), default="overall",
        help="which subsets to run every analysis in: 'overall' (default) is "
             "the whole design, 'topic' the 3 topics separately, "
             "'topic-package' the 9 topic x package cells")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="collect this run under DIR/<grouping>/ instead of the default "
             f"location. Without it, 'overall' overwrites {STATS_OUTPUT_DIR.name}/"
             "effect_size_report.txt, the published whole-design result, and "
             f"the groupings go to {GROUPED_STATS_DIR.name}/. Pass "
             "--output-dir to keep all three runs together and leave the "
             "published result untouched.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="rewrite the text reports of a finished run from its saved "
             "all_subsets.json instead of recomputing. Use after a change to "
             "the report layout; it cannot pick up a change to a statistic.")
    args = parser.parse_args()

    if args.rebuild:
        base = args.output_dir if args.output_dir is not None else GROUPED_STATS_DIR
        out_dir = rebuild_grouping(args.group_by, base)
        print(f"rebuilt {out_dir} from all_subsets.json (nothing recomputed)")
        return

    if args.output_dir is None and args.group_by == "overall":
        # the long-standing behaviour: regenerate the published report in place
        text = render(gather())
        print(text)

        ensure_output_dirs()
        (STATS_OUTPUT_DIR / "effect_size_report.txt").write_text(text, encoding="utf-8")
        print(f"wrote {STATS_OUTPUT_DIR / 'effect_size_report.txt'}")
        return

    base = args.output_dir if args.output_dir is not None else GROUPED_STATS_DIR
    out_dir = run_grouping(args.group_by, base)
    print(f"wrote {len(GROUPINGS[args.group_by])} report(s) in {out_dir}")


if __name__ == "__main__":
    main()
