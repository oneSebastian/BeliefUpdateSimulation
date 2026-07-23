"""Tests for the comment-ranking convention conversion.

Models emit an ordering (rank_i = id of the comment placed i-th); humans supply
per-slot ranks (rank_N = rank given to the comment in slot N). The two are
inverse permutations and agree on 4 of the 6 permutations of {1,2,3}, so the
cases that actually discriminate are 231 and 312.
"""

import itertools

import pandas as pd
import pytest

from belief_update_sim.ranking import ranks_from_model_ordering

NULL = {"rank_1": None, "rank_2": None, "rank_3": None}


def test_model_ordering_is_inverted_to_per_slot_ranks():
    # model placed comment 2 first, comment 3 second, comment 1 third
    row = {"rank_1": 2, "rank_2": 3, "rank_3": 1}
    assert ranks_from_model_ordering(row) == {
        "rank_1": 3,   # comment 1 was placed third
        "rank_2": 1,   # comment 2 was placed first
        "rank_3": 2,   # comment 3 was placed second
    }


def test_the_other_discriminating_permutation():
    row = {"rank_1": 3, "rank_2": 1, "rank_3": 2}
    assert ranks_from_model_ordering(row) == {"rank_1": 2, "rank_2": 3, "rank_3": 1}


@pytest.mark.parametrize("ordering", [[1, 2, 3], [2, 1, 3], [1, 3, 2], [3, 2, 1]])
def test_self_inverse_permutations_are_unchanged(ordering):
    """These four are their own inverse -- which is why the bug hid so well."""
    row = {f"rank_{i}": v for i, v in enumerate(ordering, start=1)}
    assert ranks_from_model_ordering(row) == row


def test_conversion_is_an_involution():
    """Applying the inversion twice returns the original ordering."""
    for perm in itertools.permutations([1, 2, 3]):
        row = {f"rank_{i}": v for i, v in enumerate(perm, start=1)}
        once = ranks_from_model_ordering(row)
        assert ranks_from_model_ordering(once) == row


def test_output_is_always_a_valid_ranking():
    for perm in itertools.permutations([1, 2, 3]):
        row = {f"rank_{i}": v for i, v in enumerate(perm, start=1)}
        out = ranks_from_model_ordering(row)
        assert sorted(out.values()) == [1, 2, 3]


def test_key_order_is_fixed_regardless_of_input():
    """Field order must not vary row to row in the exported JSONL."""
    for perm in itertools.permutations([1, 2, 3]):
        row = {f"rank_{i}": v for i, v in enumerate(perm, start=1)}
        assert list(ranks_from_model_ordering(row)) == ["rank_1", "rank_2", "rank_3"]


@pytest.mark.parametrize("bad", [
    {"rank_1": None, "rank_2": None, "rank_3": None},
    {"rank_1": 1, "rank_2": 1, "rank_3": 2},      # duplicate
    {"rank_1": 1, "rank_2": 2, "rank_3": None},   # incomplete
    {"rank_1": 0, "rank_2": 1, "rank_3": 2},      # out of range
    {"rank_1": 1, "rank_2": 2, "rank_3": 4},      # out of range
    {},                                            # nothing at all
])
def test_invalid_rankings_become_null_rather_than_shifting_data(bad):
    assert ranks_from_model_ordering(bad) == NULL


def test_accepts_a_pandas_series():
    """inspect_rankings passes a row of a DataFrame, not a dict."""
    series = pd.Series({"rank_1": 2.0, "rank_2": 3.0, "rank_3": 1.0})
    assert ranks_from_model_ordering(series) == {"rank_1": 3, "rank_2": 1, "rank_3": 2}


def test_nan_is_treated_as_missing():
    series = pd.Series({"rank_1": float("nan"), "rank_2": 1, "rank_3": 2})
    assert ranks_from_model_ordering(series) == NULL


def test_exporter_reuses_the_shared_helper():
    """One implementation, so the exporter and the figures cannot diverge."""
    from scripts.data import export_hf_dataset as ex
    assert ex.ranks_from_model_ordering is ranks_from_model_ordering
