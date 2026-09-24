"""Formatting lock for the combined effect-size report.

`render` is exercised on a synthetic `gather()` payload, so the layout (all six
sections, the effect-size columns) is pinned without paying for the slow
sensitivity simulation that the real `gather()` runs. The same synthetic
payload drives the subset (`--group-by`) rendering and the cross-subset
overview, and `permutation_rows` is checked against stub loaders rather than
the real workbooks.
"""

import json
import sys

import pandas as pd
import pytest

from belief_update_sim.comment_ranks import MODELS
from belief_update_sim.grouping import ALL, Group
from scripts.stats import effect_size_report as esr


def _fake_gather(group=ALL, n_groups=1, min_expected=150.0, n_comments=27):
    perm = {n: {"n": 1173, "mean_diff": 0.1, "sd_diff": 1.0, "d_z": 0.1, "p": 0.01}
            for n in MODELS}
    chi = {n: {"n_human": 1173, "n_model": 1173, "chi2": 50.0, "dof": 4,
               "p_value": 1e-9, "cramers_v": 0.15, "min_expected": min_expected}
           for n in MODELS}
    bf_delta = {n: {"n_model": 1173, "sd_model": 0.5, "sd_ratio": 1.7,
                    "variance_ratio": 2.9, "W": 30.0, "p_value": 1e-8,
                    "human_more_variable": True} for n in MODELS}
    bf_rank = {"n_comments": n_comments, "sd_human": 0.098,
               "models": {n: {"sd": 0.6, "sd_ratio": 6.1, "variance_ratio": 37.0,
                              "W": 35.0, "p_value": 1e-7, "significant": True}
                          for n in MODELS}}
    tau = {n: {"n": 1173, "mean": -0.02, "sd": 0.63, "se": 0.018,
               "ci_low": -0.05, "ci_high": 0.01} for n in MODELS}
    sens = {"n_obs": 1173, "n_participants": 391, "n_comments": n_comments,
            "permutation_dz": {"nominal": 0.10, "independent": 0.18},
            "chi2_cramers_v": {"nominal": 0.09, "independent": 0.15},
            "brown_forsythe_belief_change_sd_ratio": {"nominal": 1.3, "independent": 1.5},
            "brown_forsythe_mean_rank_sd_ratio": {"n": n_comments, "fixed_n": 2.1},
            "kendall_tau": {"nominal": 0.065, "independent": 0.11}}
    return {"perm": perm, "chi": chi, "sd_human_delta": 0.87, "n_human_delta": 1173,
            "bf_delta": bf_delta, "bf_rank": bf_rank, "tau": tau, "sens": sens,
            "group": group, "n_groups": n_groups,
            "counts": {"n_obs": 1173, "n_participants": 391}}


# ---------------------------------------------------------------------------
# the whole-design report
# ---------------------------------------------------------------------------

def test_report_render_has_all_six_sections():
    text = esr.render(_fake_gather())
    for section in ["1. PERMUTATION", "2. CHI-SQUARED", "3. BROWN-FORSYTHE",
                    "4. BROWN-FORSYTHE", "5. KENDALL", "6. SENSITIVITY"]:
        assert section in text


def test_report_render_shows_the_effect_size_columns():
    text = esr.render(_fake_gather())
    assert "d_z" in text
    assert "CramerV" in text
    assert "var ratio" in text
    assert "mean tau" in text
    # two-sided: no one-sided p column leaks through
    assert "1-sided" not in text and "2-sided" not in text


def test_report_render_lists_every_model():
    text = esr.render(_fake_gather())
    for name in MODELS:
        assert name in text


def test_ungrouped_report_carries_no_subset_material():
    text = esr.render(_fake_gather())
    assert "SUBSET" not in text
    assert "READING THIS SUBSET" not in text
    assert "Design: 391 participants x 3 topics = 1173 observations." in text


def test_render_works_without_the_group_keys():
    """A payload from before --group-by existed still renders as the whole design."""
    payload = _fake_gather()
    for key in ("group", "n_groups", "counts"):
        payload.pop(key)
    assert "SUBSET" not in esr.render(payload)


# ---------------------------------------------------------------------------
# subset reports
# ---------------------------------------------------------------------------

def test_subset_report_names_the_subset_and_keeps_all_six_sections():
    text = esr.render(_fake_gather(Group(topic="UBI"), n_groups=3))
    assert "SUBSET: topic = UBI" in text
    for section in ["1. PERMUTATION", "2. CHI-SQUARED", "3. BROWN-FORSYTHE",
                    "4. BROWN-FORSYTHE", "5. KENDALL", "6. SENSITIVITY"]:
        assert section in text
    assert "READING THIS SUBSET -- topic = UBI" in text


def test_subset_report_offers_both_bonferroni_thresholds():
    text = esr.render(_fake_gather(Group(topic="UBI"), n_groups=3))
    assert "0.05/6" in text
    assert "0.05/18" in text          # 6 models x 3 subsets as one family


def test_subset_report_states_that_clustering_is_gone():
    text = esr.render(_fake_gather(Group(topic="UBI", package="package1"), n_groups=9))
    assert "no within-participant clustering" in text


def test_subset_caveats_flag_a_thin_chi_squared_table():
    thin = esr.subset_caveats(_fake_gather(Group(topic="UBI"), 3, min_expected=3.2))
    fat = esr.subset_caveats(_fake_gather(Group(topic="UBI"), 3, min_expected=150.0))
    assert any("chi-squared approximation is no longer reliable" in l for l in thin)
    assert not any("no longer reliable" in l for l in fat)


def test_subset_caveats_flag_the_shrunken_comment_groups():
    few = esr.subset_caveats(_fake_gather(Group(topic="UBI"), 3, n_comments=9))
    all27 = esr.subset_caveats(_fake_gather(Group(topic="UBI"), 3, n_comments=27))
    assert any("almost no power" in l for l in few)
    assert not any("almost no power" in l for l in all27)


def test_subset_caveats_quote_the_minimum_detectable_effect():
    lines = esr.subset_caveats(_fake_gather(Group(topic="UBI"), 3))
    assert any("0.1000" in l for l in lines)       # sens permutation_dz nominal


# ---------------------------------------------------------------------------
# the cross-subset overview
# ---------------------------------------------------------------------------

def _fake_results(groups):
    n = len(groups)
    return [(g, _fake_gather(g, n_groups=n)) for g in groups]


def test_overview_has_a_row_per_subset_and_a_column_per_model():
    groups = [Group(topic=t) for t in ("UBI", "weight_loss", "penalty")]
    text = esr.overview(_fake_results(groups))
    assert "CROSS-SUBSET OVERVIEW" in text
    for g in groups:
        assert g.label in text
    for name in MODELS:
        assert name[:12] in text


def test_overview_reports_every_statistic():
    groups = [Group(topic=t) for t in ("UBI", "penalty")]
    text = esr.overview(_fake_results(groups))
    for wanted in ["mean paired final-stance difference", "paired Cohen's d_z",
                   "Cramer's V", "smallest expected cell",
                   "variance ratio (human/model)", "variance ratio (model/human)",
                   "mean tau", "MDE for d_z"]:
        assert wanted in text


def test_overview_reports_the_raw_difference_not_only_the_standardised_one():
    """The permuted statistic is the raw mean difference; d_z is derived."""
    group = Group(topic="UBI")
    g = _fake_gather(group, n_groups=1)
    for name in MODELS:                      # make the two visibly different
        g["perm"][name] = dict(g["perm"][name], mean_diff=0.1234, d_z=0.5678)
    text = esr.overview([(group, g)])
    raw = text.split("paired Cohen's d_z")[0]
    assert "+0.1234" in raw
    assert "+0.5678" in text.split("paired Cohen's d_z")[1]
    assert "raw scale points" in raw


def test_overview_marks_both_significance_thresholds():
    groups = [Group(topic=t) for t in ("UBI", "penalty")]
    text = esr.overview(_fake_results(groups))
    assert "0.05/6" in text and "0.05/12" in text
    assert "**" in text               # the fake p-values clear the stricter alpha


def _perm_overview(mean_diff, p):
    """The two permutation blocks of an overview whose every cell holds (mean_diff, p).

    Two subsets, so the two Bonferroni thresholds differ: * is p < 0.05/6 and
    ** is p < 0.05/12.
    """
    results = []
    for group in (Group(topic="UBI"), Group(topic="penalty")):
        g = _fake_gather(group, n_groups=2)
        for name in MODELS:
            g["perm"][name] = dict(g["perm"][name], mean_diff=mean_diff, p=p)
        results.append((group, g))
    text = esr.overview(results)
    diff_block, rest = text.split("paired Cohen's d_z")
    return diff_block, rest.split("1. PERMUTATION -- p")[1]


def test_permutation_columns_are_reported_to_four_decimals():
    diff_block, p_block = _perm_overview(0.22757, 0.33871)
    assert "+0.2276" in diff_block
    assert "0.3387" in p_block


def test_a_p_of_zero_is_reported_as_a_bound_not_as_zero():
    """0 exceedances in a million resamples is not a probability of zero."""
    _, p_block = _perm_overview(0.2, 0.0)
    assert "<0.0001" in p_block
    assert "  0.0000" not in p_block


def test_a_p_that_rounds_away_is_reported_as_a_bound():
    _, p_block = _perm_overview(0.2, 0.000004)
    assert "<0.0001" in p_block


def test_a_p_that_survives_rounding_keeps_its_digits():
    _, p_block = _perm_overview(0.2, 0.00012)
    assert "0.0001" in p_block and "<0.0001" not in p_block


def test_a_mean_difference_that_rounds_away_is_reported_as_a_bound():
    diff_block, _ = _perm_overview(0.00002, 0.5)
    assert "<0.0001" in diff_block


def test_the_significance_markers_survive_the_new_formatting():
    _, p_block = _perm_overview(0.2, 0.0)
    assert "<0.0001**" in p_block
    _, mid = _perm_overview(0.2, 0.005)      # under 0.05/6, over 0.05/18
    assert "0.0050*" in mid and "0.0050**" not in mid


def test_the_four_decimal_rule_is_explained_in_the_legend():
    text = esr.overview(_fake_results([Group(topic="UBI"), Group(topic="penalty")]))
    assert "'<0.0001' means the value rounds to 0.0000" in text


def test_summary_payload_is_json_serialisable():
    group = Group(topic="UBI", package="package1")
    blob = json.dumps(esr.summary(group, _fake_gather(group, 9)), default=float)
    restored = json.loads(blob)
    assert restored["group"] == "UBI__package1"
    assert restored["topic"] == "UBI" and restored["package"] == "package1"
    assert set(restored["permutation"]) == set(MODELS)


# ---------------------------------------------------------------------------
# permutation_rows, against stub loaders
# ---------------------------------------------------------------------------

PERSONAS = ["P1", "P2", "P3", "P1", "P2", "P3"]
STUB_TOPICS = ["UBI"] * 3 + ["penalty"] * 3
STUB_PACKAGES = ("package1", "package2", "package2") * 2


def _stub_frames(model_packages):
    human = pd.DataFrame({
        "persona_id": PERSONAS, "topic": STUB_TOPICS,
        "package": list(STUB_PACKAGES),
        "new_belief": [1.0, -1.0, 2.0, 2.0, 1.0, -2.0],
    })
    model = pd.DataFrame({
        "persona_id": PERSONAS, "topic": STUB_TOPICS,
        "package": list(model_packages),
        "new_belief": [0.0, -1.0, 0.0, 1.0, -1.0, -1.0],
    })
    return human, model                     # paired differences: 1, 0, 2, 1, 2, -1


@pytest.fixture
def stubbed(monkeypatch):
    def install(model_packages=STUB_PACKAGES):
        human, model = _stub_frames(list(model_packages))
        monkeypatch.setattr(esr, "MODELS", {"Stub": "stub.xlsx"})
        monkeypatch.setattr(esr, "load_human_normalized_data", lambda: human)
        monkeypatch.setattr(esr, "load_normalized_data", lambda path: model)
        # the real test is a million resamples; the arithmetic around it is
        # what this exercises
        monkeypatch.setattr(esr, "fast_permutation_test",
                            lambda a, b: (float((a - b).mean()), 0.5, False))
    return install


def test_permutation_rows_uses_every_pair_when_ungrouped(stubbed):
    stubbed()
    rows = esr.permutation_rows()
    assert rows["Stub"]["n"] == 6
    assert rows["Stub"]["mean_diff"] == pytest.approx(5 / 6)


def test_permutation_rows_restricts_to_the_group(stubbed):
    stubbed()
    rows = esr.permutation_rows(Group(topic="UBI"))
    assert rows["Stub"]["n"] == 3
    assert rows["Stub"]["mean_diff"] == pytest.approx(1.0)


def test_permutation_rows_restricts_to_a_package_cell(stubbed):
    stubbed()
    rows = esr.permutation_rows(Group(topic="penalty", package="package2"))
    assert rows["Stub"]["n"] == 2
    assert rows["Stub"]["mean_diff"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# what a run leaves on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def cheap_run(monkeypatch):
    """Drive run_grouping()/main() without computing anything."""
    monkeypatch.setattr(esr, "gather",
                        lambda group=ALL, n_groups=1: _fake_gather(group, n_groups))


def test_a_grouping_writes_one_report_per_subset(cheap_run, tmp_path):
    out = esr.run_grouping("topic", tmp_path)
    assert out == tmp_path / "by_topic"
    assert {p.name for p in out.iterdir()} == {
        "UBI.txt", "weight_loss.txt", "penalty.txt",
        "overview.txt", "all_subsets.txt", "all_subsets.json"}


def test_the_nine_cells_land_in_their_own_directory(cheap_run, tmp_path):
    out = esr.run_grouping("topic-package", tmp_path)
    assert out == tmp_path / "by_topic_package"
    assert len(list(out.glob("*__package*.txt"))) == 9


def test_a_single_subset_skips_the_duplicate_files(cheap_run, tmp_path):
    """An overview of one subset would just repeat that subset's report."""
    out = esr.run_grouping("overall", tmp_path)
    assert out == tmp_path / "whole_design"
    assert {p.name for p in out.iterdir()} == {"overall.txt", "all_subsets.json"}


def test_the_groupings_do_not_collide_in_one_output_dir(cheap_run, tmp_path):
    dirs = [esr.run_grouping(name, tmp_path)
            for name in ("overall", "topic", "topic-package")]
    assert len(set(dirs)) == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "by_topic", "by_topic_package", "whole_design"]


def test_output_dir_keeps_the_published_report_untouched(cheap_run, tmp_path,
                                                         monkeypatch):
    published = tmp_path / "stats"
    published.mkdir()
    (published / "effect_size_report.txt").write_text("PUBLISHED", encoding="utf-8")
    monkeypatch.setattr(esr, "STATS_OUTPUT_DIR", published)
    monkeypatch.setattr(sys, "argv", ["effect_size_report", "--group-by", "overall",
                                      "--output-dir", str(tmp_path / "grouped")])

    esr.main()

    assert (published / "effect_size_report.txt").read_text() == "PUBLISHED"
    assert (tmp_path / "grouped" / "whole_design" / "overall.txt").exists()


def test_without_output_dir_the_whole_design_still_regenerates_in_place(
        cheap_run, tmp_path, monkeypatch):
    published = tmp_path / "stats"
    published.mkdir()
    monkeypatch.setattr(esr, "STATS_OUTPUT_DIR", published)
    monkeypatch.setattr(esr, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(sys, "argv", ["effect_size_report"])

    esr.main()

    assert "EFFECT SIZES" in (published / "effect_size_report.txt").read_text()


def test_rebuild_reproduces_the_reports_without_recomputing(cheap_run, tmp_path,
                                                            monkeypatch):
    esr.run_grouping("topic", tmp_path)
    out = tmp_path / "by_topic"
    before = (out / "all_subsets.txt").read_text(encoding="utf-8")
    stamp = (out / "all_subsets.json").stat().st_mtime_ns

    # gather() would raise if the rebuild tried to recompute anything
    def explode(*args, **kwargs):
        raise AssertionError("rebuild must not recompute")

    monkeypatch.setattr(esr, "gather", explode)
    esr.rebuild_grouping("topic", tmp_path)

    assert (out / "all_subsets.txt").read_text(encoding="utf-8") == before
    assert (out / "all_subsets.json").stat().st_mtime_ns == stamp   # left alone


def test_rebuild_says_what_to_do_when_there_is_nothing_to_rebuild(tmp_path):
    with pytest.raises(SystemExit, match="run --group-by topic without --rebuild"):
        esr.rebuild_grouping("topic", tmp_path)


def test_payload_from_summary_round_trips_every_field(cheap_run, tmp_path):
    esr.run_grouping("topic", tmp_path)
    blob = json.loads(
        (tmp_path / "by_topic" / "all_subsets.json").read_text(encoding="utf-8"))
    results = esr.payload_from_summary(blob)

    assert [group.label for group, _ in results] == ["UBI", "weight_loss", "penalty"]
    for group, g in results:
        assert g["n_groups"] == 3
        assert g["group"] == group
        assert set(g["perm"]) == set(MODELS)
        # render() reads all of these; a missing key would only show up here
        assert esr.render(g) == esr.render(_fake_gather(group, 3))


def test_permutation_rows_rejects_a_package_disagreement(stubbed):
    """A cell must compare the same stimuli on both sides, or it means nothing."""
    disagreeing = ("package1", "package3", "package2") + STUB_PACKAGES[3:]
    stubbed(disagreeing)
    with pytest.raises(ValueError, match="different comment package"):
        esr.permutation_rows(Group(topic="UBI"))
