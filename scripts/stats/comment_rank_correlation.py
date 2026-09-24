"""Kendall rank correlation between each human's and their LLM twin's comment ranking.

Every human-LLM pair sharing a persona saw the same three comments, in the same
slots, for each topic. For one pair:

    tau = (N_c - N_d) / N

over the three unordered comment pairs, so tau takes one of {1, 1/3, -1/3, -1}.
1 means identical orderings, -1 exactly opposite, and values near 0 mean the two
rankings are unrelated. The reported figure is the mean over all pairs.

Reads the source files directly: results/*.xlsx for the models and the merged
human CSV. Both index their rank columns by *shown slot*, so the comparison is
slot by slot -- but the two record different things in those columns, and the
model side is converted first:

    human   rank_message_N_shown  = rank given to the comment in slot N
    model   rank_N                = id of the slot placed N-th

See belief_update_sim.ranking for why, and why mixing them silently produces
plausible but wrong numbers.

    python -m scripts.stats.comment_rank_correlation
"""

import argparse
import collections
import itertools
import json
import math

from belief_update_sim.comment_ranks import (HUMAN, MODELS, SLOTS, load_human_ranks,
                                             load_model_ranks)
from belief_update_sim.config import RESULTS_DIR, STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.grouping import ALL


def ranks_of(row):
    """{slot: rank} from rank_1/2/3, or None if missing or malformed."""
    ranks = [row["rank_1"], row["rank_2"], row["rank_3"]]
    if any(r is None for r in ranks) or sorted(ranks) != [1, 2, 3]:
        return None
    return dict(zip(SLOTS, ranks))


def kendall_tau(a, b):
    """(N_c - N_d) / N over the three unordered slot pairs."""
    concordant = discordant = 0
    for i, j in itertools.combinations(SLOTS, 2):
        if (a[i] - a[j]) * (b[i] - b[j]) > 0:
            concordant += 1
        else:
            discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def summarize(taus):
    n = len(taus)
    mean = sum(taus) / n
    var = sum((t - mean) ** 2 for t in taus) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "se": se,
            "ci_low": mean - 1.96 * se, "ci_high": mean + 1.96 * se}


def compute(group=ALL):
    human = load_human_ranks()
    per_model, per_model_topic = {}, {}
    skipped = collections.Counter()
    mismatched = collections.Counter()

    for name, filename in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise SystemExit(f"missing results file: {path}")
        model = load_model_ranks(path)

        taus, by_topic = [], collections.defaultdict(list)
        for key, h in human.items():
            # out-of-group pairs are not "skipped" -- they are not in the
            # population this run is about, so they are not counted anywhere
            if not group.matches(key[1], h["package"]):
                continue
            m = model.get(key)
            if m is None:
                skipped[name] += 1
                continue
            if h["ranks"] is None or m["ranks"] is None:
                skipped[name] += 1
                continue
            # tau pairs rankings slot by slot; different stimuli in those slots
            # would make the comparison meaningless rather than merely noisy
            if m["shown"] is not None and h["shown"] != m["shown"]:
                mismatched[name] += 1
                continue
            tau = kendall_tau(h["ranks"], m["ranks"])
            taus.append(tau)
            by_topic[key[1]].append(tau)

        per_model[name] = taus
        per_model_topic[name] = by_topic

    return per_model, per_model_topic, skipped, mismatched


def render(per_model, per_model_topic, skipped, mismatched):
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 92)
    out("Kendall rank correlation between human and LLM comment rankings")
    out("tau = (N_c - N_d) / N per human-LLM pair; possible values 1, 1/3, -1/3, -1")
    out("model rankings converted from ordering to per-slot ranks on load")
    out("=" * 92)
    if skipped:
        out(f"pairs skipped for missing or malformed rankings: {dict(skipped)}")
    if mismatched:
        out(f"pairs skipped because the shown comments differed: {dict(mismatched)}")
    out()

    out(f"{'model':<26}{'n':>6}{'mean tau':>11}{'sd':>8}{'95% CI':>20}")
    out("-" * 92)
    stats = {}
    for name, taus in per_model.items():
        s = summarize(taus)
        stats[name] = s
        ci = f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]"
        out(f"{name:<26}{s['n']:>6}{s['mean']:>+11.5f}{s['sd']:>8.4f}{ci:>20}")
    out("-" * 92)
    out()

    # a mean near zero can arise either from agreement being rare or from
    # agreement and disagreement cancelling; these are different claims
    out("distribution of tau (% of pairs)")
    values = [1.0, 1 / 3, -1 / 3, -1.0]
    out(f"{'model':<26}" + "".join(f"{v:>+10.2f}" for v in values))
    out("-" * 92)
    for name, taus in per_model.items():
        counts = collections.Counter(round(t, 4) for t in taus)
        row = f"{name:<26}"
        for v in values:
            row += f"{100 * counts[round(v, 4)] / len(taus):>10.1f}"
        out(row)
    out("-" * 92)
    out("chance baseline for independent rankings: 16.7 / 33.3 / 33.3 / 16.7")
    out(f"sd under independence: sqrt(11/27) = {math.sqrt(11 / 27):.4f}")
    out()

    topics = sorted({t for by in per_model_topic.values() for t in by})
    out("mean tau by topic")
    out(f"{'model':<26}" + "".join(f"{t:>16}" for t in topics))
    out("-" * 92)
    for name, by_topic in per_model_topic.items():
        row = f"{name:<26}"
        for topic in topics:
            taus = by_topic.get(topic, [])
            row += f"{(sum(taus) / len(taus)):>+16.5f}" if taus else f"{'-':>16}"
        out(row)
    out("-" * 92)

    return "\n".join(lines) + "\n", stats


def main():
    argparse.ArgumentParser().parse_args()

    per_model, per_model_topic, skipped, mismatched = compute()
    if not any(per_model.values()):
        raise SystemExit("no human-LLM pairs found")

    text, stats = render(per_model, per_model_topic, skipped, mismatched)

    ensure_output_dirs()
    (STATS_OUTPUT_DIR / "comment_rank_correlation.txt").write_text(text, encoding="utf-8")
    (STATS_OUTPUT_DIR / "comment_rank_correlation.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {STATS_OUTPUT_DIR / 'comment_rank_correlation.txt'}")
    print(f"wrote {STATS_OUTPUT_DIR / 'comment_rank_correlation.json'}")


if __name__ == "__main__":
    main()
