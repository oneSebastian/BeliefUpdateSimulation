"""The `group=` plumbing through the five test modules the report assembles.

Each module is exercised on stub loaders or on the human CSV alone, so these
stay fast: the point is that a subset reaches the statistic, not that scipy
still works. The whole-design numbers themselves are locked in
`test_stats_reproductions.py`.
"""

import numpy as np
import pytest

from belief_update_sim.config import RESULTS_DIR
from belief_update_sim.grouping import ALL, GROUPINGS, Group
from scripts.stats import belief_change_variability as bcv
from scripts.stats import comment_rank_correlation as crc
from scripts.stats import comment_rank_variability as crv
from scripts.stats import post_stance_distribution as psd
from scripts.stats import power_sensitivity as ps

# any file that really exists, so the modules' own existence check passes
EXISTING = "merged_llm_participants_data_with_normalized_beliefs.csv"


# ---------------------------------------------------------------------------
# comment_rank_variability: which of the 27 comments a subset holds
# ---------------------------------------------------------------------------

def test_expected_comment_counts_follow_the_design():
    assert crv.expected_comments(ALL) == 27
    assert crv.expected_comments(Group(topic="UBI")) == 9
    assert crv.expected_comments(Group(topic="UBI", package="package1")) == 3


KEYS = [(topic, package, msg)
        for topic in ("UBI", "weight_loss", "penalty")
        for package in ("package1", "package2", "package3")
        for msg in (1, 2, 3)]


def test_comments_in_group_keeps_everything_for_all():
    assert crv.comments_in_group(KEYS, ALL) == KEYS


def test_comments_in_group_selects_a_topic():
    out = crv.comments_in_group(KEYS, Group(topic="UBI"))
    assert len(out) == 9
    assert {k[0] for k in out} == {"UBI"}


def test_comments_in_group_selects_a_cell():
    out = crv.comments_in_group(KEYS, Group(topic="UBI", package="package2"))
    assert out == [("UBI", "package2", m) for m in (1, 2, 3)]


@pytest.mark.parametrize("name", ["topic", "topic-package"])
def test_the_subsets_of_a_grouping_partition_the_27_comments(name):
    seen = [k for g in GROUPINGS[name] for k in crv.comments_in_group(KEYS, g)]
    assert sorted(seen) == sorted(KEYS)


# ---------------------------------------------------------------------------
# belief_change_variability: real human data, one CSV read per call
# ---------------------------------------------------------------------------

def test_human_abs_delta_covers_the_whole_design_by_default():
    assert bcv.human_abs_delta().size == 1173


def test_human_abs_delta_subsets_partition_the_observations():
    sizes = [bcv.human_abs_delta(g).size for g in GROUPINGS["topic"]]
    assert sizes == [391, 391, 391]
    assert sum(sizes) == bcv.human_abs_delta().size


def test_human_abs_delta_cells_partition_the_observations():
    sizes = [bcv.human_abs_delta(g).size for g in GROUPINGS["topic-package"]]
    assert sum(sizes) == 1173


# ---------------------------------------------------------------------------
# post_stance_distribution, against stub loaders
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_stances(monkeypatch):
    """Two topics x two packages, each cell holding one of each Likert level.

    Every level is populated, because a level empty on *both* sides gives
    chi2_contingency a zero expected frequency and it refuses the table -- see
    `test_post_stance_rejects_a_level_no_one_used`.
    """
    import pandas as pd

    rows = [(f"P{i}", topic, package, level)
            for i, (topic, package, level) in enumerate(
                (t, p, l) for t in ("UBI", "penalty")
                for p in ("package1", "package2")
                for l in (-2, -1, 0, 1, 2))]
    frame = pd.DataFrame(rows, columns=["persona_id", "topic", "package",
                                        "new_belief"])
    monkeypatch.setattr(psd, "MODELS", {"Stub": EXISTING})
    monkeypatch.setattr(psd, "load_human_normalized_data", lambda: frame)
    monkeypatch.setattr(psd, "load_normalized_data", lambda path: frame)
    return frame


def test_post_stance_counts_the_whole_frame_by_default(stub_stances):
    human_counts, n_human, _ = psd.compute()
    assert n_human == 20
    assert human_counts == [4, 4, 4, 4, 4]          # SD, D, N, A, SA


def test_post_stance_counts_only_the_subset(stub_stances):
    human_counts, n_human, results = psd.compute(Group(topic="UBI"))
    assert n_human == 10
    assert human_counts == [2, 2, 2, 2, 2]
    assert results["Stub"]["n_model"] == 10


def test_post_stance_counts_only_the_cell(stub_stances):
    human_counts, n_human, _ = psd.compute(Group(topic="penalty",
                                                 package="package2"))
    assert n_human == 5
    assert human_counts == [1, 1, 1, 1, 1]


def test_post_stance_rejects_a_level_no_one_used(stub_stances, monkeypatch):
    """A subset thin enough to empty a level on both sides names the level."""
    thin = stub_stances[stub_stances["new_belief"] != 2]
    monkeypatch.setattr(psd, "load_human_normalized_data", lambda: thin)
    monkeypatch.setattr(psd, "load_normalized_data", lambda path: thin)
    with pytest.raises(ValueError, match="no observations at post-stance 2"):
        psd.compute(Group(topic="UBI"))


# ---------------------------------------------------------------------------
# comment_rank_correlation, against stub loaders
# ---------------------------------------------------------------------------

FORWARD = {1: 1, 2: 2, 3: 3}
REVERSED = {1: 3, 2: 2, 3: 1}


def _record(ranks, package):
    return {"ranks": ranks, "shown": (1, 2, 3), "package": package}


@pytest.fixture
def stub_ranks(monkeypatch):
    human = {
        ("P1", "UBI"): _record(FORWARD, "package1"),
        ("P2", "UBI"): _record(FORWARD, "package2"),
        ("P3", "penalty"): _record(FORWARD, "package1"),
    }
    model = {
        ("P1", "UBI"): _record(FORWARD, "package1"),      # tau = +1
        ("P2", "UBI"): _record(REVERSED, "package2"),     # tau = -1
        ("P3", "penalty"): _record(FORWARD, "package1"),  # tau = +1
    }
    monkeypatch.setattr(crc, "MODELS", {"Stub": EXISTING})
    monkeypatch.setattr(crc, "load_human_ranks", lambda: human)
    monkeypatch.setattr(crc, "load_model_ranks", lambda path: model)


def test_rank_correlation_uses_every_pair_by_default(stub_ranks):
    per_model, per_model_topic, _, _ = crc.compute()
    assert sorted(per_model["Stub"]) == [-1.0, 1.0, 1.0]
    assert set(per_model_topic["Stub"]) == {"UBI", "penalty"}


def test_rank_correlation_restricts_to_the_topic(stub_ranks):
    per_model, per_model_topic, _, _ = crc.compute(Group(topic="UBI"))
    assert sorted(per_model["Stub"]) == [-1.0, 1.0]
    assert set(per_model_topic["Stub"]) == {"UBI"}


def test_rank_correlation_restricts_to_the_cell(stub_ranks):
    per_model, _, _, _ = crc.compute(Group(topic="UBI", package="package1"))
    assert per_model["Stub"] == [1.0]


def test_out_of_group_pairs_are_not_counted_as_skipped(stub_ranks):
    """`skipped` means malformed data; being in another cell is not a defect."""
    _, _, skipped, mismatched = crc.compute(Group(topic="UBI", package="package1"))
    assert dict(skipped) == {} and dict(mismatched) == {}


# ---------------------------------------------------------------------------
# power_sensitivity: MDEs re-solved at the subset's own n
# ---------------------------------------------------------------------------

def test_a_smaller_subset_can_only_detect_a_larger_effect():
    cell_n = 130
    assert ps.paired_dz_mde(cell_n) > ps.paired_dz_mde(391) > ps.paired_dz_mde(1173)


@pytest.fixture
def cheap_sensitivity(monkeypatch):
    """Replace the slow Brown-Forsythe simulation with a call recorder."""
    calls = []

    def fake_mde(base, n, human_more_variable, rng):
        calls.append(n)
        return 1.5

    def fake_collect(group=ALL):
        keys = list(range(crv.expected_comments(group)))
        return {ps.HUMAN: {k: 2.0 for k in keys}}, keys

    monkeypatch.setattr(ps, "bf_ratio_mde", fake_mde)
    monkeypatch.setattr(ps, "human_abs_delta",
                        lambda group=ALL: np.array([0.0, 1.0, 2.0, 1.0]))
    monkeypatch.setattr(ps.crv, "collect", fake_collect)
    return calls


def test_sensitivity_solves_at_the_subsets_own_sample_size(cheap_sensitivity):
    r = ps.compute(Group(topic="UBI"))
    assert r["group"] == "UBI"
    assert r["n_obs"] == 391 and r["n_participants"] == 391
    assert r["n_comments"] == 9
    assert r["brown_forsythe_mean_rank_sd_ratio"]["n"] == 9
    assert r["permutation_dz"]["nominal"] == pytest.approx(ps.paired_dz_mde(391))


def test_the_clustered_simulation_is_not_repeated_inside_a_cell(cheap_sensitivity):
    """nominal n == independent n there, so solving it twice is wasted work."""
    ps.compute(Group(topic="UBI"))
    assert cheap_sensitivity == [391, 9]          # |delta| once, mean-ranks once


def test_the_whole_design_still_solves_both_sample_sizes(cheap_sensitivity):
    ps.compute(ALL)
    assert cheap_sensitivity == [1173, 391, 27]
