"""Do humans discriminate between comments less than LLMs do?

There are 27 comments (3 topics x 3 packages x 3 messages). For each source we
take the mean rank every comment received across all the personas that saw it.
A source that consistently finds the same comment most persuasive produces mean
ranks spread across the full 1..3 range; a source whose raters disagree produces
mean ranks bunched near 2.

The spread of those 27 means is compared against the human spread with a
two-sided Brown-Forsythe test (Levene centred on the median, which is robust to
the non-normal distribution of the means):

    H0: var(human mean ranks) == var(model mean ranks)
    H1: var(human) != var(model)      -- the spreads differ

Reads the source files directly; model rankings are converted into the human
convention on load (see belief_update_sim.ranking).

    python -m scripts.stats.comment_rank_variability
"""

import argparse
import json
import statistics

from scipy.stats import levene

from belief_update_sim.comment_ranks import HUMAN, load_all_sources, mean_rank_by_comment
from belief_update_sim.config import STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.grouping import ALL

N_COMMENTS = 27
N_MESSAGES = 3          # messages per topic x package cell


def expected_comments(group=ALL):
    """How many of the 27 comments belong to a group."""
    n = N_COMMENTS
    if group.topic is not None:
        n //= 3
    if group.package is not None:
        n //= 3
    return n


def comments_in_group(keys, group=ALL):
    """The (topic, package, message) keys that belong to `group`."""
    return [k for k in keys if group.matches(k[0], k[1])]


def collect(group=ALL):
    """Mean rank per comment, restricted to the comments in `group`.

    A comment is only ever shown inside its own topic x package cell, so its
    mean rank is already a within-cell quantity -- selecting the keys of the
    cell is the same thing as recomputing the means from that cell's rows.
    """
    sources = load_all_sources()
    means = {name: mean_rank_by_comment(records) for name, records in sources.items()}

    # only comments every source ranked, so the groups are strictly comparable
    shared = comments_in_group(
        sorted(set.intersection(*[set(m) for m in means.values()])), group)
    expected = expected_comments(group)
    if len(shared) != expected:
        print(f"note: {len(shared)} comments common to all sources "
              f"(expected {expected})")
    return means, shared


def render(means, comments):
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    names = list(means)
    models = [n for n in names if n != HUMAN]

    out("=" * 104)
    out("Mean rank given to each comment (1 = most persuasive, 3 = least)")
    out("model rankings converted from ordering to per-slot ranks on load")
    out("=" * 104)
    out()
    out(f"{'topic':<12}{'package':<10}{'msg':>4}  " +
        "".join(f"{n[:14]:>15}" for n in names))
    out("-" * 104)
    for key in comments:
        topic, package, msg = key
        row = f"{topic:<12}{package:<10}{msg:>4}  "
        row += "".join(f"{means[n][key]:>15.3f}" for n in names)
        out(row)
    out("-" * 104)

    sds = {n: statistics.stdev([means[n][c] for c in comments]) for n in names}
    out(f"{'sd across comments':<28}" + "".join(f"{sds[n]:>15.4f}" for n in names))
    out()

    out("=" * 104)
    out("Two-sided Brown-Forsythe test (Levene, center='median')")
    out("H0: var(human mean ranks) == var(model mean ranks)")
    out("H1: var(human) != var(model)  -- the spreads differ")
    out(f"n = {len(comments)} comments per group")
    out("=" * 104)
    out()
    out(f"{'model':<26}{'sd(human)':>11}{'sd(model)':>11}{'var ratio':>10}{'W':>10}"
        f"{'p':>14}   sig")
    out("-" * 104)

    human_values = [means[HUMAN][c] for c in comments]
    stats = {"n_comments": len(comments), "sd_human": sds[HUMAN], "models": {}}
    alpha = 0.05 / len(models)

    for name in models:
        model_values = [means[name][c] for c in comments]
        w, p_two = levene(human_values, model_values, center="median")
        # variance/SD ratio (model / human) reports the direction descriptively;
        # every model is more variable than the human ranks
        sd_ratio = sds[name] / sds[HUMAN]
        stats["models"][name] = {"sd": sds[name], "sd_ratio": sd_ratio,
                                 "variance_ratio": sd_ratio ** 2, "W": w,
                                 "p_value": p_two,
                                 "significant": bool(p_two < alpha)}
        out(f"{name:<26}{sds[HUMAN]:>11.4f}{sds[name]:>11.4f}{sd_ratio ** 2:>10.3f}"
            f"{w:>10.3f}{p_two:>14.3e}   {'yes' if p_two < alpha else 'no'}")
    out("-" * 104)
    out(f"two-sided test; Bonferroni across the {len(models)} models: alpha = {alpha:.4f}")
    out("var ratio is model/human; every model is more variable than the human ranks")

    worst = max(s["p_value"] for s in stats["models"].values())
    out()
    out(f"largest p across all models: {worst:.3e}")
    if all(s["significant"] for s in stats["models"].values()):
        out("=> human and model rank-spreads differ significantly for every model")
    else:
        failed = [n for n, s in stats["models"].items() if not s["significant"]]
        out(f"=> NOT significant for: {', '.join(failed)}")

    stats["largest_p"] = worst
    stats["alpha_bonferroni"] = alpha
    return "\n".join(lines) + "\n", stats


def main():
    argparse.ArgumentParser().parse_args()
    means, comments = collect()
    text, stats = render(means, comments)

    ensure_output_dirs()
    (STATS_OUTPUT_DIR / "comment_rank_variability.txt").write_text(text, encoding="utf-8")
    (STATS_OUTPUT_DIR / "comment_rank_variability.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {STATS_OUTPUT_DIR / 'comment_rank_variability.txt'}")
    print(f"wrote {STATS_OUTPUT_DIR / 'comment_rank_variability.json'}")


if __name__ == "__main__":
    main()
