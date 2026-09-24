"""Tests for the design cells the statistical tests can be restricted to.

The `Group` semantics are checked on synthetic frames; `design_counts` is
checked against the real human data, because the property that matters for the
subset reports -- one observation per participant inside a cell, so the nominal
and independent sample sizes coincide -- is a fact about the study, not about
the code.
"""

import pandas as pd
import pytest

from belief_update_sim.grouping import (ALL, GROUPINGS, PACKAGES, TOPICS, Group,
                                        design_counts)


def frame():
    return pd.DataFrame({
        "persona_id": ["P1", "P1", "P2", "P2"],
        "topic": ["UBI", "penalty", "UBI", "penalty"],
        "package": ["package1", "package2", "package3", "package1"],
        "value": [1, 2, 3, 4],
    })


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------

def test_all_is_the_unrestricted_group():
    assert ALL.is_all
    assert ALL.label == "overall"
    assert not Group(topic="UBI").is_all


def test_labels_are_filename_safe_and_distinct():
    labels = [g.label for g in GROUPINGS["topic-package"]]
    assert len(set(labels)) == 9
    assert all(set(l) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
               for l in labels)
    assert Group(topic="UBI", package="package1").label == "UBI__package1"


def test_title_names_the_constraints():
    assert Group(topic="UBI").title == "topic = UBI"
    assert "package1" in Group(topic="UBI", package="package1").title
    assert ALL.title == "all topics and packages"


def test_all_matches_every_record():
    assert ALL.matches("UBI", "package1")
    assert ALL.matches("penalty", None)


def test_topic_group_ignores_the_package():
    g = Group(topic="UBI")
    assert g.matches("UBI", "package3")
    assert g.matches("UBI")
    assert not g.matches("penalty", "package3")


def test_cell_group_requires_both():
    g = Group(topic="UBI", package="package1")
    assert g.matches("UBI", "package1")
    assert not g.matches("UBI", "package2")
    assert not g.matches("penalty", "package1")
    # a record that does not carry a package cannot be in a package cell
    assert not g.matches("UBI")


def test_filter_returns_the_frame_unchanged_for_all():
    df = frame()
    assert ALL.filter(df) is df


def test_filter_selects_the_topic():
    out = Group(topic="UBI").filter(frame())
    assert list(out["value"]) == [1, 3]


def test_filter_selects_the_cell():
    out = Group(topic="UBI", package="package3").filter(frame())
    assert list(out["value"]) == [3]


def test_filter_refuses_to_return_an_empty_subset():
    """Silently testing zero rows would surface much later, inside scipy."""
    with pytest.raises(ValueError, match="no rows for"):
        Group(topic="not_a_topic").filter(frame())


# ---------------------------------------------------------------------------
# the grouping registry
# ---------------------------------------------------------------------------

def test_groupings_cover_the_design():
    assert [g.label for g in GROUPINGS["overall"]] == ["overall"]
    assert len(GROUPINGS["topic"]) == 3
    assert len(GROUPINGS["topic-package"]) == 9
    assert {g.topic for g in GROUPINGS["topic"]} == set(TOPICS)
    assert {g.package for g in GROUPINGS["topic-package"]} == set(PACKAGES)


def test_every_row_falls_in_exactly_one_group_of_a_grouping():
    df = frame()
    for name in ("topic", "topic-package"):
        hits = [sum(g.matches(r.topic, r.package) for g in GROUPINGS[name])
                for r in df.itertuples()]
        assert hits == [1] * len(df)


# ---------------------------------------------------------------------------
# design_counts, against the real data
# ---------------------------------------------------------------------------

def test_design_counts_reproduce_the_whole_design():
    assert design_counts(ALL) == {"n_obs": 1173, "n_participants": 391}


def test_topic_subsets_partition_the_observations():
    counts = [design_counts(g)["n_obs"] for g in GROUPINGS["topic"]]
    assert counts == [391, 391, 391]
    assert sum(counts) == design_counts(ALL)["n_obs"]


def test_package_cells_partition_the_observations():
    counts = [design_counts(g)["n_obs"] for g in GROUPINGS["topic-package"]]
    assert sum(counts) == design_counts(ALL)["n_obs"]
    assert all(100 < n < 160 for n in counts)       # ~130 per cell


@pytest.mark.parametrize("name", ["topic", "topic-package"])
def test_inside_a_cell_there_is_no_clustering_left(name):
    """One observation per participant, so nominal n == independent n."""
    for group in GROUPINGS[name]:
        c = design_counts(group)
        assert c["n_obs"] == c["n_participants"], group.label
