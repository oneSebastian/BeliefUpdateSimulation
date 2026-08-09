"""Locks on the sensitivity power analysis (minimum detectable effects).

The closed-form MDEs (paired d_z, Cramer's V, Kendall tau) are cheap and pinned
directly. The Brown-Forsythe simulation is expensive, so only its cheap
invariants are checked (monotonic power, a ratio above 1), not a full solve.
"""

import math

import numpy as np
import pytest

from scripts.stats import power_sensitivity as ps


def test_more_data_detects_smaller_paired_effects():
    assert ps.paired_dz_mde(ps.N_OBS) < ps.paired_dz_mde(ps.N_PARTICIPANTS)


def test_paired_dz_mde_matches_the_reported_value():
    assert ps.paired_dz_mde(ps.N_OBS) == pytest.approx(0.102, abs=0.003)
    assert ps.paired_dz_mde(ps.N_PARTICIPANTS) == pytest.approx(0.177, abs=0.003)


def test_cramers_v_mde_matches_the_reported_value():
    assert ps.cramers_v_mde(2 * ps.N_OBS) == pytest.approx(0.086, abs=0.003)
    assert ps.cramers_v_mde(2 * ps.N_PARTICIPANTS) == pytest.approx(0.149, abs=0.003)


def test_kendall_mde_is_the_paired_dz_scaled_by_tau_spread():
    n = ps.N_OBS
    assert ps.kendall_tau_mde(n) == pytest.approx(
        ps.paired_dz_mde(n) * math.sqrt(11 / 27))


def test_bf_power_increases_with_the_ratio():
    base = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4], dtype=float)
    rng = np.random.default_rng(0)
    low = ps._bf_power(base, 200, 1.1, True, rng, n_sims=300)
    rng = np.random.default_rng(0)
    high = ps._bf_power(base, 200, 2.0, True, rng, n_sims=300)
    assert high > low
    assert 0.0 <= low <= high <= 1.0


def test_null_ratio_of_one_stays_near_alpha():
    base = np.array([0, 0, 1, 1, 1, 2, 2, 3, 4], dtype=float)
    rng = np.random.default_rng(1)
    # at ratio 1 the groups are exchangeable: rejection rate ~ alpha, well below 0.8
    assert ps._bf_power(base, 200, 1.0, True, rng, n_sims=500) < 0.1
