"""Regression tests for the NaN guard in belief_update_sim/permutation_test.py.

A single unparsed response used to yield p = 0.0 rather than an error: the
observed statistic skipped the NaN (pandas) while the null distribution
propagated it (numpy), so no resample could ever exceed the observed value.
This produced a spurious "significant" result for gemini-3-flash-preview.
"""

import numpy as np
import pandas as pd
import pytest

from belief_update_sim.permutation_test import (
    _reject_nulls,
    fast_permutation_test,
    perform_permutation_test,
)

CLEAN_A = pd.Series([1.0, -1.0, 2.0, 0.0, -2.0, 1.0, 1.0, -1.0])
CLEAN_B = pd.Series([0.0, -1.0, 1.0, 1.0, -1.0, 0.0, 2.0, -2.0])


def _with_nan(series, position=3):
    out = series.copy()
    out.iloc[position] = np.nan
    return out


def test_guard_accepts_clean_samples():
    _reject_nulls(CLEAN_A, CLEAN_B)


def test_guard_rejects_nan_in_first_sample():
    with pytest.raises(ValueError, match="NaN"):
        _reject_nulls(_with_nan(CLEAN_A), CLEAN_B)


def test_guard_rejects_nan_in_second_sample():
    with pytest.raises(ValueError, match="NaN"):
        _reject_nulls(CLEAN_A, _with_nan(CLEAN_B))


def test_guard_reports_how_many_nans_per_sample():
    a = CLEAN_A.copy()
    a.iloc[0] = a.iloc[1] = np.nan
    with pytest.raises(ValueError, match=r"2 in sample_1, 1 in sample_2"):
        _reject_nulls(a, _with_nan(CLEAN_B))


def test_guard_includes_context_when_given():
    with pytest.raises(ValueError, match="for a.xlsx vs b.xlsx"):
        _reject_nulls(_with_nan(CLEAN_A), CLEAN_B, context=" for a.xlsx vs b.xlsx")


def test_guard_accepts_plain_numpy_arrays():
    _reject_nulls(CLEAN_A.to_numpy(), CLEAN_B.to_numpy())


def test_perform_permutation_test_raises_instead_of_returning_p_zero():
    """The whole point: this used to return (obs, 0.0, True)."""
    with pytest.raises(ValueError, match="NaN"):
        perform_permutation_test(_with_nan(CLEAN_A), CLEAN_B)


def test_fast_permutation_test_raises_instead_of_returning_p_zero():
    with pytest.raises(ValueError, match="NaN"):
        fast_permutation_test(_with_nan(CLEAN_A), CLEAN_B)


def test_fast_permutation_test_still_runs_on_clean_data():
    obs, p, reject = fast_permutation_test(CLEAN_A, CLEAN_B)
    assert obs == pytest.approx(float(np.mean(CLEAN_A - CLEAN_B)))
    assert 0.0 <= p <= 1.0
    assert reject == (p < 0.0167)


def test_identical_samples_give_zero_effect_and_no_rejection():
    obs, p, reject = fast_permutation_test(CLEAN_A, CLEAN_A.copy())
    assert obs == 0.0
    assert p == 1.0
    assert not reject
