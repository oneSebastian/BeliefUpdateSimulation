import time
import itertools
import json
import string
import re
import unicodedata
import pandas as pd
import numpy as np
from scipy.stats import permutation_test

from .data_loading import load_normalized_data, load_human_normalized_data


def _reject_nulls(sample_1, sample_2, context=""):
    """Refuse to run on samples containing NaN.

    A NaN here does not fail, it silently manufactures significance. The
    observed statistic is computed on pandas Series, where np.mean dispatches
    to Series.mean() and skips NaN, so it looks healthy. scipy then computes
    the null distribution on numpy arrays, where np.mean propagates NaN, so
    every permuted statistic is NaN. `abs(null) >= abs(obs)` is False against
    NaN for every resample, the exceedance count stays 0, and p comes out
    exactly 0.0 -- reported as the most significant result possible.

    This produced a spurious p = 0.0 for gemini-3-flash-preview (one unparsed
    response out of 1173); the correct value is p ~ 0.52.
    """
    n1 = int(np.isnan(np.asarray(sample_1, dtype=float)).sum())
    n2 = int(np.isnan(np.asarray(sample_2, dtype=float)).sum())
    if n1 or n2:
        raise ValueError(
            f"permutation test received NaN values{context}: "
            f"{n1} in sample_1, {n2} in sample_2. "
            "Drop or impute these pairs explicitly -- leaving them in yields "
            "p = 0.0 rather than an error."
        )


def perform_permutation_test(sample_1, sample_2):
    _reject_nulls(sample_1, sample_2)

    def statistic(x, y):
        return np.mean(x - y)
    observed_diff = np.mean(sample_1 - sample_2)
    # Perform paired permutation test
    N_TOTAL = 1_000_000
    CHUNK_SIZE = 100_000
    N_CHUNKS = N_TOTAL // CHUNK_SIZE
    count = 0
    obs = statistic(sample_1, sample_2)
    rng = np.random.default_rng(42)
    for _ in range(N_CHUNKS):
        res = permutation_test(
            (sample_1, sample_2),
            statistic,
            permutation_type="samples",random_state=rng,
            n_resamples=CHUNK_SIZE
        )
        count += np.sum(np.abs(res.null_distribution) >= np.abs(obs))
    p_value = count / N_TOTAL
    alpha = 0.0167
    reject = (p_value < alpha)
    return observed_diff, p_value, reject


def fast_permutation_test(sample_1, sample_2):
    _reject_nulls(sample_1, sample_2)
    d = (sample_1 - sample_2).to_numpy()
    obs = np.mean(d)

    N_TOTAL = 1_000_000
    CHUNK_SIZE = 100_000
    N_CHUNKS = N_TOTAL // CHUNK_SIZE

    rng = np.random.default_rng(42)
    count = 0

    for _ in range(N_CHUNKS):
        signs = rng.integers(0, 2, size=(CHUNK_SIZE, len(d))) * 2 - 1
        perm_means = np.mean(signs * d, axis=1)
        count += np.sum(np.abs(perm_means) >= np.abs(obs))

    p_value = count / N_TOTAL
    alpha = 0.0167
    reject = p_value < alpha

    return obs, p_value, reject


def cohens_dz(sample_1, sample_2):
    """Paired Cohen's d_z = mean(diff) / sd(diff), the effect size that pairs
    with the paired permutation test. NaN pairs are dropped first."""
    d = np.asarray(sample_1, dtype=float) - np.asarray(sample_2, dtype=float)
    d = d[~np.isnan(d)]
    return float(np.mean(d) / np.std(d, ddof=1))


def permutation_test_on_paths(path1, path2):
    paths = [path1, path2]
    data = []
    for path in paths:
        if str(path).endswith('merged_llm_participants_data_with_normalized_beliefs.csv'):
            data.append(load_human_normalized_data())
        else:
            data.append(load_normalized_data(path))
    merged_df = pd.merge(
        data[0],
        data[1],
        on=['persona_id', 'topic'],
        how='inner',
        suffixes=('_0', '_1'),
        validate='one_to_one',
        indicator=True
    )
    if (merged_df['_merge'] != 'both').any():
        raise ValueError("Not all rows have a matching partner")

    # validate='one_to_one' checks cardinality, not completeness: a row whose
    # response never parsed merges fine and arrives here as NaN.
    _reject_nulls(
        merged_df['new_belief_0'],
        merged_df['new_belief_1'],
        context=f" for {path1} vs {path2}",
    )

    return perform_permutation_test(merged_df['new_belief_0'], merged_df['new_belief_1'])
    #return fast_permutation_test(merged_df['new_belief_0'], merged_df['new_belief_1'])
