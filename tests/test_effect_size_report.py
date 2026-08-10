"""Formatting lock for the combined effect-size report.

`render` is exercised on a synthetic `gather()` payload, so the layout (all six
sections, the effect-size columns) is pinned without paying for the slow
sensitivity simulation that the real `gather()` runs.
"""

from belief_update_sim.comment_ranks import MODELS
from scripts.stats import effect_size_report as esr


def _fake_gather():
    perm = {n: {"n": 1173, "mean_diff": 0.1, "sd_diff": 1.0, "d_z": 0.1, "p": 0.01}
            for n in MODELS}
    chi = {n: {"n_human": 1173, "n_model": 1173, "chi2": 50.0, "dof": 4,
               "p_value": 1e-9, "cramers_v": 0.15, "min_expected": 150.0}
           for n in MODELS}
    bf_delta = {n: {"n_model": 1173, "sd_model": 0.5, "sd_ratio": 1.7,
                    "variance_ratio": 2.9, "W": 30.0, "p_value": 1e-8,
                    "human_more_variable": True} for n in MODELS}
    bf_rank = {"n_comments": 27, "sd_human": 0.098,
               "models": {n: {"sd": 0.6, "sd_ratio": 6.1, "variance_ratio": 37.0,
                              "W": 35.0, "p_value": 1e-7, "significant": True}
                          for n in MODELS}}
    tau = {n: {"n": 1173, "mean": -0.02, "sd": 0.63, "se": 0.018,
               "ci_low": -0.05, "ci_high": 0.01} for n in MODELS}
    sens = {"permutation_dz": {"nominal": 0.10, "independent": 0.18},
            "chi2_cramers_v": {"nominal": 0.09, "independent": 0.15},
            "brown_forsythe_belief_change_sd_ratio": {"nominal": 1.3, "independent": 1.5},
            "brown_forsythe_mean_rank_sd_ratio": {"fixed_27": 2.1},
            "kendall_tau": {"nominal": 0.065, "independent": 0.11}}
    return {"perm": perm, "chi": chi, "sd_human_delta": 0.87, "n_human_delta": 1173,
            "bf_delta": bf_delta, "bf_rank": bf_rank, "tau": tau, "sens": sens}


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
