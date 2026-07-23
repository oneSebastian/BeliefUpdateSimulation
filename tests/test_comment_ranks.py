"""Tests for the shared comment-rank loading and per-comment aggregation."""

import pytest

from belief_update_sim import comment_ranks as cr


def record(ranks, shown, package="package1"):
    return {"ranks": ranks, "shown": shown, "package": package}


def test_as_slot_ranks_accepts_a_permutation():
    assert cr._as_slot_ranks([3, 1, 2]) == {1: 3, 2: 1, 3: 2}


@pytest.mark.parametrize("bad", [[1, 1, 2], [1, 2, None], [0, 1, 2], [1, 2, 4], []])
def test_as_slot_ranks_rejects_anything_else(bad):
    assert cr._as_slot_ranks(bad) is None


@pytest.mark.parametrize("value,expected", [
    ("[2, 1, 3]", (2, 1, 3)),
    ([3, 2, 1], (3, 2, 1)),
    ("not a list", None),
    (None, None),
])
def test_perm_parses_message_order(value, expected):
    assert cr._perm(value) == expected


# ---------------------------------------------------------------------------
# per-comment aggregation
# ---------------------------------------------------------------------------

def test_rank_is_attributed_to_the_comment_that_occupied_the_slot():
    """The whole point: ranks are per slot, means are per source comment."""
    records = {
        ("P0001", "UBI"): record({1: 1, 2: 2, 3: 3}, (3, 1, 2)),
    }
    means = cr.mean_rank_by_comment(records)
    # slot 1 held source comment 3 and got rank 1
    assert means[("UBI", "package1", 3)] == 1
    assert means[("UBI", "package1", 1)] == 2
    assert means[("UBI", "package1", 2)] == 3


def test_means_average_across_participants():
    records = {
        ("P0001", "UBI"): record({1: 1, 2: 2, 3: 3}, (1, 2, 3)),
        ("P0002", "UBI"): record({1: 3, 2: 2, 3: 1}, (1, 2, 3)),
    }
    means = cr.mean_rank_by_comment(records)
    assert means[("UBI", "package1", 1)] == 2.0
    assert means[("UBI", "package1", 2)] == 2.0
    assert means[("UBI", "package1", 3)] == 2.0


def test_different_orderings_still_aggregate_onto_the_same_comment():
    """Two participants saw comment 2 in different slots; both ranked it first."""
    records = {
        ("P0001", "UBI"): record({1: 1, 2: 2, 3: 3}, (2, 1, 3)),
        ("P0002", "UBI"): record({1: 2, 2: 1, 3: 3}, (1, 2, 3)),
    }
    means = cr.mean_rank_by_comment(records)
    assert means[("UBI", "package1", 2)] == 1.0


def test_rows_without_a_usable_ranking_are_skipped():
    records = {
        ("P0001", "UBI"): record(None, (1, 2, 3)),
        ("P0002", "UBI"): record({1: 1, 2: 2, 3: 3}, None),
        ("P0003", "UBI"): record({1: 1, 2: 2, 3: 3}, (1, 2, 3)),
    }
    means = cr.mean_rank_by_comment(records)
    assert means == {("UBI", "package1", 1): 1,
                     ("UBI", "package1", 2): 2,
                     ("UBI", "package1", 3): 3}


def test_packages_and_topics_do_not_collide():
    records = {
        ("P0001", "UBI"): record({1: 1, 2: 2, 3: 3}, (1, 2, 3), "package1"),
        ("P0002", "UBI"): record({1: 3, 2: 2, 3: 1}, (1, 2, 3), "package2"),
        ("P0003", "penalty"): record({1: 2, 2: 1, 3: 3}, (1, 2, 3), "package1"),
    }
    means = cr.mean_rank_by_comment(records)
    assert means[("UBI", "package1", 1)] == 1
    assert means[("UBI", "package2", 1)] == 3
    assert means[("penalty", "package1", 1)] == 2


def test_every_registered_model_file_exists():
    from belief_update_sim.config import RESULTS_DIR
    missing = [f for f in cr.MODELS.values() if not (RESULTS_DIR / f).exists()]
    assert missing == []
