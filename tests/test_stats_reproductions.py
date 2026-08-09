"""Regression locks on the reproduced distribution- and variability-level results.

These pin the published numbers to the source data: the human absolute-belief-
change SD (0.87), the one-sided Brown-Forsythe p-values, and the chi-squared
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


def test_one_sided_p_guards_the_wrong_direction():
    # replicate the rule used in compute(): halve only when the direction matches
    p_two = 0.02
    assert (p_two / 2 if True else 1 - p_two / 2) == 0.01      # human larger
    assert (p_two / 2 if False else 1 - p_two / 2) == 0.99     # human smaller


# ---------------------------------------------------------------------------
# reproductions against the real source files
# ---------------------------------------------------------------------------

def test_human_absolute_change_sd_matches_the_paper():
    sd_human, n_human, _ = bcv.compute()
    assert n_human == 1173
    assert sd_human == pytest.approx(0.87, abs=0.005)


def test_belief_change_variability_is_one_sided_and_significant():
    sd_human, _, results = bcv.compute()
    assert set(results) == set(bcv.MODELS)
    for name, r in results.items():
        assert r["human_more_variable"] is True
        # one-sided p is exactly half the two-sided p when the direction holds
        assert r["p_one_sided"] == pytest.approx(r["p_two_sided"] / 2)
        assert r["p_one_sided"] < 0.0083            # Bonferroni across 6 models
    # Claude is the weakest; the paper reports p = 0.002 (one-sided)
    assert results["Claude-Opus-4.6"]["p_one_sided"] == pytest.approx(0.002, abs=5e-4)


def test_post_stance_chi_squared_reproduces_all_significant():
    human_counts, n_human, results = psd.compute()
    assert n_human == 1173
    assert sum(human_counts) == 1173
    for name, r in results.items():
        assert r["dof"] == 4
        assert r["min_expected"] >= 5          # chi-squared approximation valid
        assert r["p_value"] < 1e-4             # "p < 0.0001 for all LLMs"
