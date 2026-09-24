"""Sensitivity power analysis: the smallest effect each test could detect.

For every hypothesis test in the paper we fix power = 0.80 and the significance
threshold at the Bonferroni-corrected alpha = 0.05 / 6 = 0.0083 used throughout,
then solve for the *minimum detectable effect* (MDE) the sample supports.

Because the design is nested (391 participants x 3 topics = 1173 observations),
each MDE is reported at two sample sizes:

    nominal      the n the test was actually run on (observation level)
    independent  the number of independent participants (391)

The gap between the two is the price of the within-participant clustering.

Effect-size metric per test:

    permutation (paired mean stance difference)   paired Cohen's d_z
    chi-squared (2 x 5 post-stance distribution)  Cramer's V   (= Cohen's w here)
    Brown-Forsythe (|belief change| spread)       SD ratio / variance ratio
    Brown-Forsythe (27 comment mean-rank spread)  SD ratio / variance ratio
    Kendall tau (comment convincingness)          detectable mean tau

The two closed-form tests (paired t and chi-squared) use statsmodels power
solvers. The two Brown-Forsythe tests have no closed-form power, so the MDE is
found by simulation: the empirical human distribution is resampled and one
group's spread is scaled until a two-sided median-centred Levene test rejects at
80% power. Seeded, so the numbers are reproducible.

``compute()`` takes a :class:`belief_update_sim.grouping.Group`, which is how
``scripts.stats.effect_size_report --group-by`` re-solves the MDEs at a topic's
or a topic x package cell's own sample size. Inside a cell the nesting is gone
-- one observation per participant -- so the nominal and independent n coincide
there and the simulation is run once rather than twice.

    python -m scripts.stats.power_sensitivity
"""

import argparse
import json
import math

import numpy as np
from scipy.stats import levene
from statsmodels.stats.power import GofChisquarePower, TTestPower

from belief_update_sim.comment_ranks import HUMAN
from belief_update_sim.config import STATS_OUTPUT_DIR, ensure_output_dirs
from belief_update_sim.grouping import ALL, design_counts
from scripts.stats import comment_rank_variability as crv
from scripts.stats.belief_change_variability import human_abs_delta

ALPHA = 0.05 / 6            # 0.00833, the correction used everywhere in the paper
POWER = 0.80

# The whole-design sample sizes. compute() reads the sizes it actually uses
# from the data (see belief_update_sim.grouping.design_counts), so that a
# subset run solves at its own n; for the ungrouped run they are these.
N_OBS = 1173               # persona x topic observations
N_PARTICIPANTS = 391       # independent participants

TAU_SD_NULL = math.sqrt(11 / 27)   # sd of the 3-item tau under independence (0.638)

SEED = 20240609
N_SIMS = 4000              # simulation draws per power evaluation


# ---------------------------------------------------------------------------
# closed-form MDEs
# ---------------------------------------------------------------------------

def paired_dz_mde(n):
    """Minimum detectable paired Cohen's d_z for a two-sided paired test."""
    return float(TTestPower().solve_power(
        nobs=n, alpha=ALPHA, power=POWER, alternative="two-sided"))


def cramers_v_mde(n_total):
    """Minimum detectable Cramer's V for a 2 x 5 chi-squared (df = 4).

    statsmodels solves for Cohen's w; for a table with two rows
    Cramer's V = w / sqrt(min(r, c) - 1) = w, so the returned value *is* V.
    """
    w = float(GofChisquarePower().solve_power(
        nobs=n_total, alpha=ALPHA, power=POWER, n_bins=5))
    return w


def kendall_tau_mde(n, sd_ref=TAU_SD_NULL):
    """Detectable mean |tau|: the paired-test d_z MDE scaled by tau's spread."""
    dz = float(TTestPower().solve_power(
        nobs=n, alpha=ALPHA, power=POWER, alternative="two-sided"))
    return dz * sd_ref


# ---------------------------------------------------------------------------
# simulation MDE for the Brown-Forsythe variance tests
# ---------------------------------------------------------------------------

def _bf_power(base, n, ratio, human_more_variable, rng, n_sims=N_SIMS):
    """Two-sided median-centred Levene power at a given SD ratio.

    `base` is the empirical reference distribution (resampled for both groups).
    The second group's deviations from the median are scaled so that the SD
    ratio between the more- and less-variable group is `ratio` (>= 1).
    `human_more_variable` sets which group carries the larger spread; the test
    itself is non-directional.
    """
    med = float(np.median(base))
    hits = 0
    for _ in range(n_sims):
        a = rng.choice(base, size=n)               # "human" group
        b0 = rng.choice(base, size=n)
        if human_more_variable:                    # shrink the model group
            b = med + (b0 - med) / ratio
        else:                                      # inflate the model group
            b = med + (b0 - med) * ratio
        _, p_two = levene(a, b, center="median")
        if p_two < ALPHA:
            hits += 1
    return hits / n_sims


def bf_ratio_mde(base, n, human_more_variable, rng):
    """Smallest SD ratio the Levene test detects at 80% power (bisection)."""
    lo, hi = 1.0, 2.0
    # expand the bracket until the upper end is powerful enough
    while _bf_power(base, n, hi, human_more_variable, rng) < POWER:
        lo, hi = hi, hi * 1.5
        if hi > 50:
            return float("nan")
    for _ in range(22):
        mid = (lo + hi) / 2
        if _bf_power(base, n, mid, human_more_variable, rng) < POWER:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def compute(group=ALL):
    """MDEs at the sample sizes `group` actually supplies.

    For the whole design these are the module constants (1173 / 391 / 27). For
    a subset they are read from the data: inside a cell each participant
    contributes one observation, so the nominal and independent n coincide and
    the expensive Brown-Forsythe simulation is run once instead of twice.
    """
    rng = np.random.default_rng(SEED)

    counts = design_counts(group)
    n_obs, n_participants = counts["n_obs"], counts["n_participants"]

    delta_base = human_abs_delta(group)
    means, comments = crv.collect(group)
    rank_base = np.array([means[HUMAN][k] for k in comments], dtype=float)
    n_comments = len(comments)

    bf_nominal = bf_ratio_mde(delta_base, n_obs, True, rng)
    bf_delta = {
        "nominal": bf_nominal,
        "independent": (bf_nominal if n_participants == n_obs
                        else bf_ratio_mde(delta_base, n_participants, True, rng)),
    }
    bf_rank = {
        "n": n_comments,
        "fixed_n": bf_ratio_mde(rank_base, n_comments, False, rng),
    }

    return {
        "alpha": ALPHA, "power": POWER,
        "group": group.label,
        "n_obs": n_obs, "n_participants": n_participants,
        "n_comments": n_comments,
        "permutation_dz": {
            "nominal": paired_dz_mde(n_obs),
            "independent": paired_dz_mde(n_participants),
        },
        "chi2_cramers_v": {
            "nominal": cramers_v_mde(2 * n_obs),
            "independent": cramers_v_mde(2 * n_participants),
        },
        "brown_forsythe_belief_change_sd_ratio": bf_delta,
        "brown_forsythe_mean_rank_sd_ratio": bf_rank,
        "kendall_tau": {
            "nominal": kendall_tau_mde(n_obs),
            "independent": kendall_tau_mde(n_participants),
        },
    }


def render(r):
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    out("=" * 84)
    out("Sensitivity power analysis: minimum detectable effect at 80% power")
    out(f"alpha = {r['alpha']:.5f} (Bonferroni 0.05/6), power = {r['power']:.2f}")
    out(f"nominal n = {r['n_obs']} observations; "
        f"independent n = {r['n_participants']} participants")
    if r["group"] != "overall":
        out(f"subset: {r['group']}")
    out("=" * 84)
    out()

    def two(d, fmt):
        n = d.get("nominal")
        i = d.get("independent")
        return f"{fmt(n):>16}{fmt(i):>16}" if n is not None else f"{fmt(i):>16}"

    out(f"{'test / effect size':<44}{'nominal n':>16}{'independent':>16}")
    out("-" * 84)

    p = r["permutation_dz"]
    out(f"{'permutation  paired Cohen d_z':<44}"
        f"{p['nominal']:>16.4f}{p['independent']:>16.4f}")

    c = r["chi2_cramers_v"]
    out(f"{'chi-squared  Cramer V':<44}"
        f"{c['nominal']:>16.4f}{c['independent']:>16.4f}")

    b = r["brown_forsythe_belief_change_sd_ratio"]
    out(f"{'Brown-Forsythe |belief change|  SD ratio':<44}"
        f"{b['nominal']:>16.3f}{b['independent']:>16.3f}")
    out(f"{'  (equivalently, variance ratio)':<44}"
        f"{b['nominal']**2:>16.3f}{b['independent']**2:>16.3f}")

    rank = r["brown_forsythe_mean_rank_sd_ratio"]
    out(f"{'Brown-Forsythe mean-rank spread  SD ratio':<44}"
        f"{'n = ' + str(rank['n']) + ':':>16}{rank['fixed_n']:>16.3f}")

    k = r["kendall_tau"]
    out(f"{'Kendall  detectable mean |tau|':<44}"
        f"{k['nominal']:>16.4f}{k['independent']:>16.4f}")
    out("-" * 84)
    out("d_z, Cramer V and mean |tau| are the smallest effects detectable;")
    out("SD ratio is how many times more (or less) variable one group must be.")
    return "\n".join(lines) + "\n"


def main():
    argparse.ArgumentParser().parse_args()
    r = compute()
    text = render(r)

    ensure_output_dirs()
    (STATS_OUTPUT_DIR / "power_sensitivity.txt").write_text(text, encoding="utf-8")
    (STATS_OUTPUT_DIR / "power_sensitivity.json").write_text(
        json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nwrote {STATS_OUTPUT_DIR / 'power_sensitivity.txt'}")
    print(f"wrote {STATS_OUTPUT_DIR / 'power_sensitivity.json'}")


if __name__ == "__main__":
    main()
