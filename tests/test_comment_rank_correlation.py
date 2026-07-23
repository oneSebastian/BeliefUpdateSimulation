"""Tests for the human-LLM comment ranking correlation."""

import pytest

from scripts.stats import comment_rank_correlation as crc


def ranks(a, b, c):
    return {1: a, 2: b, 3: c}


def test_identical_rankings_give_plus_one():
    assert crc.kendall_tau(ranks(1, 2, 3), ranks(1, 2, 3)) == 1.0


def test_reversed_rankings_give_minus_one():
    assert crc.kendall_tau(ranks(1, 2, 3), ranks(3, 2, 1)) == -1.0


def test_one_swap_gives_one_third():
    # swapping the two adjacent ranks flips exactly one of the three pairs
    assert crc.kendall_tau(ranks(1, 2, 3), ranks(2, 1, 3)) == pytest.approx(1 / 3)


def test_two_discordant_pairs_give_minus_one_third():
    assert crc.kendall_tau(ranks(1, 2, 3), ranks(3, 1, 2)) == pytest.approx(-1 / 3)


def test_tau_is_symmetric():
    a, b = ranks(2, 3, 1), ranks(1, 3, 2)
    assert crc.kendall_tau(a, b) == crc.kendall_tau(b, a)


@pytest.mark.parametrize("perm", [(1, 2, 3), (1, 3, 2), (2, 1, 3),
                                  (2, 3, 1), (3, 1, 2), (3, 2, 1)])
def test_only_four_values_are_attainable(perm):
    tau = crc.kendall_tau(ranks(1, 2, 3), ranks(*perm))
    assert any(tau == pytest.approx(v) for v in (1.0, 1 / 3, -1 / 3, -1.0))


def test_mean_over_all_permutations_is_zero():
    """A model ranking independently of the human averages to tau = 0."""
    import itertools
    taus = [crc.kendall_tau(ranks(1, 2, 3), ranks(*p))
            for p in itertools.permutations((1, 2, 3))]
    assert sum(taus) / len(taus) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# rank extraction
# ---------------------------------------------------------------------------

def test_ranks_of_reads_a_valid_row():
    assert crc.ranks_of({"rank_1": 3, "rank_2": 1, "rank_3": 2}) == {1: 3, 2: 1, 3: 2}


@pytest.mark.parametrize("bad", [
    {"rank_1": None, "rank_2": 1, "rank_3": 2},
    {"rank_1": 1, "rank_2": 1, "rank_3": 2},
    {"rank_1": 0, "rank_2": 1, "rank_3": 2},
])
def test_malformed_rankings_are_rejected(bad):
    assert crc.ranks_of(bad) is None


def test_summarize_reports_mean_and_interval():
    s = crc.summarize([1.0, 1 / 3, -1 / 3, -1.0])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.0)
    assert s["ci_low"] < 0 < s["ci_high"]
