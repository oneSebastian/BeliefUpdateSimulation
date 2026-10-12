"""Tests for the LLM ranking analysis (scripts/stats/model_ranking.py)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance

from scripts.stats import model_ranking as mr

TOPICS = ("UBI", "penalty", "weight_loss")


def _human(n_participants=60, seed=0):
    """Every participant answers all three topics; initial beliefs cycle through
    the scale so each topic x package x init cell holds several people."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_participants):
        for t, topic in enumerate(TOPICS):
            init = (i + t) % 5 - 2
            final = int(np.clip(init + rng.integers(-1, 2), -2, 2))
            rows.append({"persona_id": f"P{i:03d}", "topic": topic, "package": "package1",
                         "init_stance": init, "new_belief": final})
    return pd.DataFrame(rows)


def _model(human, new_belief):
    return pd.DataFrame({"persona_id": human["persona_id"], "topic": human["topic"],
                         "package": human["package"], "new_belief": new_belief})


def _frame(seed=0):
    human = _human(seed=seed)
    y = human["new_belief"].to_numpy()
    models = {"Perfect": _model(human, y), "Off1": _model(human, y + 1),
              "Off2": _model(human, y + 2)}
    return mr.assemble(human, models)[0]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def test_loo_cell_median_excludes_own_case():
    frame = pd.DataFrame({"topic": "t", "package": "p", "init": 0,
                          "human": [0, 1, 2, 5]})
    assert mr.loo_cell_median(frame).tolist() == [2, 2, 1, 1]


def test_loo_cell_median_rejects_single_case_cell():
    frame = pd.DataFrame({"topic": ["t", "t", "t"], "package": "p",
                          "init": [0, 0, 1], "human": [0, 1, 2]})
    with pytest.raises(ValueError, match="single case"):
        mr.loo_cell_median(frame)


def test_assemble_aligns_models_by_key_and_drops_unparsed():
    human = _human(n_participants=10)
    y = human["new_belief"].to_numpy(dtype=float)
    shuffled = _model(human, y).sample(frac=1, random_state=1)
    broken = _model(human, y).astype({"new_belief": float})
    broken.loc[3, "new_belief"] = np.nan

    frame, dropped = mr.assemble(human, {"A": shuffled, "B": broken})
    assert dropped == {"A": 0, "B": 1}
    assert len(frame) == len(human) - 1
    assert (frame["A"] == frame["human"]).all()      # re-aligned despite the shuffle
    assert (frame[mr.NO_CHANGE] == frame["init"]).all()
    assert frame[mr.CELL_MEDIAN].notna().all()


def test_assemble_rejects_missing_cases_and_package_mismatch():
    human = _human(n_participants=10)
    model = _model(human, human["new_belief"])
    with pytest.raises(ValueError, match="missing"):
        mr.assemble(human, {"A": model.iloc[1:]})
    model.loc[0, "package"] = "package2"
    with pytest.raises(ValueError, match="package"):
        mr.assemble(human, {"A": model})


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _random_data(n=200, s=3, seed=2):
    rng = np.random.default_rng(seed)
    h = rng.integers(-2, 3, n).astype(float)
    init = rng.integers(-2, 3, n).astype(float)
    p = rng.integers(-2, 3, (n, s)).astype(float)
    p[:, 1] = rng.integers(-4, 5, n) / 2                # half-point values too
    return h, p, init


def test_score_with_unit_weights_matches_direct_computation():
    h, p, init = _random_data()
    res = {k: v[0] for k, v in mr.score(np.ones((1, len(h))), h, p, init).items()}
    for j in range(p.shape[1]):
        assert res["mae"][j] == pytest.approx(np.abs(h - p[:, j]).mean())
        assert res["bias"][j] == pytest.approx((h - p[:, j]).mean())
        assert res["wasserstein"][j] == pytest.approx(wasserstein_distance(h, p[:, j]))
        ratio = np.abs(p[:, j] - init).std() / np.abs(h - init).std()
        assert res["spread"][j] == pytest.approx(abs(np.log(ratio)))


def test_score_weight_two_equals_duplicating_the_case():
    h, p, init = _random_data(n=50)
    weights = np.ones((1, len(h)))
    weights[0, :5] = 2
    weighted = mr.score(weights, h, p, init)
    dup = np.r_[np.arange(len(h)), np.arange(5)]
    plain = mr.score(np.ones((1, len(dup))), h[dup], p[dup], init[dup])
    for k in weighted:
        np.testing.assert_allclose(weighted[k], plain[k])


def test_score_no_change_has_infinite_spread():
    h, p, init = _random_data(n=50)
    p[:, 0] = init
    assert np.isinf(mr.score(np.ones((1, len(h))), h, p, init)["spread"][0, 0])


def test_bootstrap_is_seeded_and_shaped():
    frame = _frame()
    a = mr.bootstrap(frame, ["Perfect", "Off1"], n_boot=30, seed=5)
    b = mr.bootstrap(frame, ["Perfect", "Off1"], n_boot=30, seed=5)
    assert a["mae"].shape == (30, 2)
    np.testing.assert_array_equal(a["mae"], b["mae"])
    # constant offsets have no sampling variability
    np.testing.assert_allclose(a["mae"][:, 1], 1.0)


# --------------------------------------------------------------------------
# pairwise tests
# --------------------------------------------------------------------------

def test_cluster_sign_flip_flips_whole_participants():
    rng = np.random.default_rng(3)
    per_participant = rng.normal(0.2, 1, (40, 1))
    codes = np.arange(40)
    once = mr.cluster_sign_flip(per_participant, codes, n_perm=5000, seed=1)
    # the same participants with three identical cases each: a row-level test
    # would see 3x the evidence, the cluster test must not
    thrice = mr.cluster_sign_flip(np.repeat(per_participant, 3, axis=0),
                                  np.repeat(codes, 3), n_perm=5000, seed=1)
    assert thrice[0] == pytest.approx(once[0])
    assert thrice[1] == pytest.approx(once[1])


def test_cluster_sign_flip_null_and_shift():
    rng = np.random.default_rng(4)
    codes = np.repeat(np.arange(100), 3)
    zero = np.zeros((300, 1))
    shift = rng.normal(1, 1, (300, 1))
    assert mr.cluster_sign_flip(zero, codes, n_perm=2000)[1][0] == 1.0
    assert mr.cluster_sign_flip(shift, codes, n_perm=2000)[1][0] == pytest.approx(1 / 2001)


def test_letter_display_shares_letters_only_between_non_different_sources():
    names = ["a", "b", "c", "d"]
    significant = {("a", "c"), ("a", "d"), ("b", "d")}
    pairs = [{"a": x, "b": y, "significant": (x, y) in significant}
             for i, x in enumerate(names) for y in names[i + 1:]]
    assert mr.letter_display(names, pairs) == {"a": "a", "b": "ab", "c": "bc", "d": "c"}


def test_letter_display_all_equal_is_one_tier():
    pairs = [{"a": "x", "b": "y", "significant": False}]
    assert mr.letter_display(["x", "y"], pairs) == {"x": "a", "y": "a"}


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def result():
    return mr.compute(_frame(), ["Off2", "Perfect", "Off1"], n_boot=200, n_perm=2000)


def test_compute_ranks_by_distance_from_humans(result):
    mae = result["metrics"]["mae"]
    assert [mae[n]["rank"] for n in ("Perfect", "Off1", "Off2")] == [1, 2, 3]
    assert mae["Perfect"]["estimate"] == 0.0
    assert mae["Off1"]["estimate"] == pytest.approx(1.0)
    assert mae["Perfect"]["p_best"] == 1.0
    assert mae["Perfect"]["rank_ci"] == [1, 1]
    assert "rank" not in mae[mr.NO_CHANGE]                # baselines are not ranked
    assert result["metrics"]["wasserstein"]["Off2"]["estimate"] == pytest.approx(2.0)
    assert result["metrics"]["bias"]["Off1"]["estimate"] == pytest.approx(-1.0)


def test_compute_orders_pairs_best_first_and_separates_tiers(result):
    assert result["order"][0] == "Perfect"
    llm_pairs = [r for r in result["pairwise"]
                 if {r["a"], r["b"]} <= {"Perfect", "Off1", "Off2"}]
    assert len(llm_pairs) == 3
    assert all(r["significant"] and r["diff"] < 0 for r in llm_pairs)
    letters = result["letters"]
    assert len({letters["Perfect"], letters["Off1"], letters["Off2"]}) == 3
    n_sources = len(result["llms"]) + len(result["baselines"])
    assert len(result["pairwise"]) == n_sources * (n_sources - 1) // 2


def test_render_lists_every_source_and_handles_inf(result):
    text = mr.render(result)
    for name in ["Perfect", "Off1", "Off2", mr.NO_CHANGE, mr.CELL_MEDIAN]:
        assert name in text
    assert "inf" in text                                  # no-change spread
    assert "Tiers on MAE" in text
    assert "Omnibus" in text


# --------------------------------------------------------------------------
# omnibus, Bonferroni, no baselines, LaTeX
# --------------------------------------------------------------------------

def test_omnibus_rejects_different_models_and_keeps_identical_ones():
    frame = _frame()
    assert mr.omnibus(frame, ["Perfect", "Off1", "Off2"], n_perm=2000)["p"] \
        == pytest.approx(1 / 2001)
    frame["Twin"] = frame["Off1"]
    same = mr.omnibus(frame, ["Off1", "Twin"], n_perm=2000)
    assert same["statistic"] == 0.0 and same["p"] == 1.0


def test_pairwise_uses_bonferroni_on_the_raw_p(monkeypatch):
    frame = _frame()
    sources = ["Perfect", "Off1", "Off2"]
    reps = np.zeros((10, 3))
    # three pairs: threshold 0.05 / 3 ~ 0.0167, so 0.02 is not significant
    monkeypatch.setattr(mr, "cluster_sign_flip",
                        lambda diffs, codes, n, s: (np.zeros(3), np.array([0.001, 0.02, 0.5])))
    rows = mr.pairwise(frame, sources, reps)
    assert [r["p"] for r in rows] == [0.001, 0.02, 0.5]
    assert [r["significant"] for r in rows] == [True, False, False]
    assert all("p_holm" not in r for r in rows)


@pytest.fixture(scope="module")
def llm_only():
    return mr.compute(_frame(), ["Off2", "Perfect", "Off1"], n_boot=100, n_perm=500,
                      baselines=())


def test_compute_without_baselines_pairs_only_the_llms(llm_only):
    assert llm_only["baselines"] == []
    assert len(llm_only["pairwise"]) == 3
    assert llm_only["alpha_pair"] == pytest.approx(0.05 / 3)
    assert mr.NO_CHANGE not in llm_only["metrics"]["mae"]
    assert mr.NO_CHANGE not in mr.render(llm_only)


def test_render_tex_bolds_only_bonferroni_significant_pairs(llm_only):
    r = dict(llm_only, pairwise=[dict(row) for row in llm_only["pairwise"]])
    r["pairwise"][0]["p"] = 0.001                        # below 0.05 / 3
    r["pairwise"][1]["p"] = 0.02                         # above it
    tex = mr.render_tex(r)
    assert r"\textbf{0.0010}" in tex
    assert "0.0200" in tex and r"\textbf{0.0200}" not in tex
    assert "0.05/3" in tex and r"\begin{tabular}" in tex
