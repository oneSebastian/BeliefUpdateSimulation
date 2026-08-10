"""Regression locks on the reproduced distribution- and variability-level results.

These pin the published numbers to the source data: the human absolute-belief-
change SD (0.87), the two-sided Brown-Forsythe p-values, and the chi-squared
post-stance comparison. If a data or loader change moves them, a test fails
rather than the paper quietly going stale.
"""

import numpy as np
import pytest

from scipy.stats import chi2_contingency, levene

from scripts.stats import belief_change_variability as bcv
from scripts.stats import post_stance_distribution as psd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_post_stance_counts_tabulates_the_five_levels():
    import pandas as pd
    s = pd.Series([-2, -2, 0, 1, 2, 2, 2])
    assert psd.counts(s) == [2, 0, 1, 1, 3]


def test_chi_squared_of_identical_distributions_is_non_significant():
    table = np.array([[100, 200, 100], [100, 200, 100]])
    _, p, _, _ = chi2_contingency(table)
    assert p == pytest.approx(1.0)


def test_abs_delta_is_invariant_to_the_pro_con_sign_flip():
    """|post - init| is unchanged when both terms flip, so raw == normalised."""
    post, init = 2, -1
    assert abs(post - init) == abs((-post) - (-init))


def test_cohens_dz_is_mean_over_sd_of_differences():
    from belief_update_sim.permutation_test import cohens_dz
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])
    d = a - b
    assert cohens_dz(a, b) == pytest.approx(d.mean() / d.std(ddof=1))


# ---------------------------------------------------------------------------
# reproductions against the real source files
# ---------------------------------------------------------------------------

def test_human_absolute_change_sd_matches_the_paper():
    sd_human, n_human, _ = bcv.compute()
    assert n_human == 1173
    assert sd_human == pytest.approx(0.87, abs=0.005)


def test_belief_change_variability_is_two_sided_and_significant():
    sd_human, _, results = bcv.compute()
    assert set(results) == set(bcv.MODELS)
    for name, r in results.items():
        assert r["human_more_variable"] is True
        assert r["p_value"] < 0.0083            # two-sided, Bonferroni across 6 models
        assert r["sd_ratio"] == pytest.approx(sd_human / r["sd_model"])
        assert r["variance_ratio"] == pytest.approx(r["sd_ratio"] ** 2)
    # Claude is the weakest; two-sided p ~ 0.0047 (twice the old one-sided 0.0024)
    assert results["Claude-Opus-4.6"]["p_value"] == pytest.approx(0.0047, abs=5e-4)


def test_comment_rank_variability_is_two_sided_and_significant():
    import contextlib
    import io

    from scripts.stats import comment_rank_variability as crv
    means, comments = crv.collect()
    with contextlib.redirect_stdout(io.StringIO()):        # render() also prints
        _, stats = crv.render(means, comments)
    assert stats["n_comments"] == 27
    for name, r in stats["models"].items():
        assert r["significant"] is True
        assert r["p_value"] < 0.0083                       # two-sided, Bonferroni
        assert r["variance_ratio"] == pytest.approx(r["sd_ratio"] ** 2)
    assert stats["largest_p"] < 0.0083


def test_post_stance_chi_squared_reproduces_all_significant():
    human_counts, n_human, results = psd.compute()
    assert n_human == 1173
    assert sum(human_counts) == 1173
    for name, r in results.items():
        assert r["dof"] == 4
        assert r["min_expected"] >= 5          # chi-squared approximation valid
        assert r["p_value"] < 1e-4             # "p < 0.0001 for all LLMs"
        n_total = r["n_human"] + r["n_model"]
        assert r["cramers_v"] == pytest.approx((r["chi2"] / n_total) ** 0.5)
