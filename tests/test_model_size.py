"""Tests for the model-size analysis and its LaTeX table.

Two modules, split so that reformatting the table does not re-run the
bootstrap: scripts.stats.model_size computes and writes a CSV,
scripts.stats.model_size_table renders that CSV. The scoring itself belongs to
scripts.stats.model_ranking and is tested in test_model_ranking.py.
"""

import functools
import json

import numpy as np
import pandas as pd
import pytest

from belief_update_sim.config import CONFIGS_DIR
from scripts.stats import model_ranking as mr
from scripts.stats import model_size as ms
from scripts.stats import model_size_table as mst

TOPICS = ("UBI", "penalty", "weight_loss")


def _human(n_participants=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_participants):
        for t, topic in enumerate(TOPICS):
            init = (i + t) % 5 - 2
            final = int(np.clip(init + rng.integers(-1, 2), -2, 2))
            rows.append({"persona_id": f"P{i:03d}", "topic": topic,
                         "package": "package1", "init_stance": init,
                         "new_belief": final})
    return pd.DataFrame(rows)


def _model(human, new_belief):
    return pd.DataFrame({"persona_id": human["persona_id"], "topic": human["topic"],
                         "package": human["package"], "new_belief": new_belief})


@functools.lru_cache(maxsize=None)
def _result(seed=0):
    """compute() over four invented models in two families, cheaply.

    Invented rather than real ones so the rendering tests do not depend on the
    sweep's registry or on MEASURED_CHECKPOINT_GIB -- they check the layout,
    not the findings. The names still follow the real convention, because
    parsing them is part of what is under test.

    Acme-30B-A2B is sparse, so the family orders 1B, 30B-A2B, 8B by active
    parameters but 1B, 8B, 30B-A2B by stored ones. The two sorts have to
    disagree, or a test that claims to check the ordering checks nothing.
    """
    human = _human(seed=seed)
    y = human["new_belief"].to_numpy()
    models = {"Acme-1B": _model(human, y + 1),
              "Acme-30B-A2B": _model(human, y),
              "Acme-8B": _model(human, y - 1),
              "Zeta-12B": _model(human, y + 2)}
    frame, dropped = mr.assemble(human, models)
    total = len(human)
    valid = {n: {"n": total - dropped[n], "total": total,
                 "share": (total - dropped[n]) / total} for n in models}
    retries = {n: {"n": i * 100, "total": total, "share": i * 100 / total,
                   "max_tries": 3}
               for i, n in enumerate(models)}
    # All three budgets are tiny on purpose. These tests check the wiring --
    # which statistic lands in which column, how it is rendered -- not the
    # precision of any estimate. n_resamples especially: fast_permutation_test
    # defaults to a million sign-flips, which is six seconds per source here.
    return ms.compute(frame, models, valid, retries,
                      n_boot=200, n_perm=500, n_resamples=2000)


def _table(seed=0):
    """A fresh frame each call -- several tests mutate it -- off the cached run."""
    return ms.to_frame(_result(seed))


# --------------------------------------------------------------------------
# model size: dense, sparse, and Gemma's effective-parameter names
# --------------------------------------------------------------------------

def test_a_dense_model_stores_what_it_activates():
    assert ms.parse_size("Qwen3.5-9B") == (9.0, 9.0)


def test_a_mixture_of_experts_activates_less_than_it_stores():
    """The A-suffix is the per-token compute; the leading number is capacity."""
    assert ms.parse_size("Qwen3.5-35B-A3B") == (35.0, 3.0)
    assert ms.parse_size("gemma-4-26B-A4B-it") == (26.0, 4.0)


def test_a_gemma_e_name_states_the_effective_size_not_the_stored_one():
    """E2B is not a 2B checkpoint. Gemma's E-naming counts parameters resident
    on the accelerator; per-layer embeddings are not, so the file is far bigger.
    The stored figure therefore comes from the measured checkpoint, not the name.
    """
    total, active = ms.parse_size("gemma-4-E2B-it")
    assert active == 2.0
    assert total == pytest.approx(5.12, abs=0.05)


def test_the_name_is_checked_against_the_served_checkpoint():
    """Every dense and MoE name matches the checkpoint it was served from; the
    two E-names deliberately do not. If that ever stops holding, the size axis
    is wrong and this is where it surfaces.
    """
    for model in ms.MEASURED_CHECKPOINT_GIB:
        measured = ms.measured_params_b(model)
        stated = ms._SIZE_RE.search(model)
        ratio = measured / float(stated.group(2))
        if stated.group(1) == "E":
            assert ratio > 1.9, f"{model}: E-name now matches its checkpoint"
        else:
            assert ratio < 1.25, f"{model}: {measured:.2f}B stored, name says {stated.group(2)}B"


def test_checkpoint_sizes_are_recorded_for_the_whole_sweep():
    """logs/ is gitignored, so these cannot be re-read at runtime."""
    names = {json.loads(p.read_text(encoding="utf-8-sig"))[0]["model"]
             for p in (CONFIGS_DIR / "model_size").glob("*.json")}
    assert names <= set(ms.MEASURED_CHECKPOINT_GIB)


def test_a_name_without_a_size_is_rejected_rather_than_sorted_arbitrarily():
    with pytest.raises(ValueError, match="no '<number>B' token"):
        ms.parse_size("mystery-model")


@pytest.mark.parametrize("model,expected", [
    ("Qwen3.5-9B", "Qwen3.5"),
    ("Qwen3.5-0.8B", "Qwen3.5"),
    ("Qwen3.5-35B-A3B", "Qwen3.5"),
    ("gemma-4-E2B-it", "gemma-4"),
    ("gemma-4-12B-it", "gemma-4"),
    ("gemma-4-26B-A4B-it", "gemma-4"),
])
def test_the_family_is_the_name_up_to_its_size_token(model, expected):
    assert ms.family(model) == expected


# --------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------

def test_the_sweep_is_ordered_by_active_parameters():
    """Compute per token, not stored weights. That is what puts the sparse
    35B-A3B down beside the 4B models, which is where its accuracy sits."""
    names = [json.loads(p.read_text(encoding="utf-8-sig"))[0]["model"]
             for p in sorted((CONFIGS_DIR / "model_size").glob("*.json"))]
    assert sorted(names, key=ms.size_key) == [
        "Qwen3.5-0.8B", "Qwen3.5-2B", "gemma-4-E2B-it", "Qwen3.5-35B-A3B",
        "Qwen3.5-4B", "gemma-4-E4B-it", "gemma-4-26B-A4B-it", "Qwen3.5-9B",
        "Qwen3.5-122B-A10B", "gemma-4-12B-it", "Qwen3.5-27B", "gemma-4-31B-it",
    ]


def test_a_sparse_model_does_not_sort_by_its_stored_size():
    """35B-A3B ahead of the 4B models, not next to the 27B."""
    assert ms.size_key("Qwen3.5-35B-A3B") < ms.size_key("Qwen3.5-4B")
    assert ms.size_key("Qwen3.5-35B-A3B") < ms.size_key("Qwen3.5-27B")


def test_equal_active_sizes_fall_back_to_stored_size_then_name():
    """gemma-4-E4B-it and gemma-4-26B-A4B-it both activate 4B; the one holding
    26B sorts after the one holding 8B."""
    assert ms.size_key("gemma-4-E4B-it") < ms.size_key("gemma-4-26B-A4B-it")


# --------------------------------------------------------------------------
# the registry, read from the sweep's configs
# --------------------------------------------------------------------------

def test_the_registry_comes_from_the_configs_that_are_actually_submitted():
    entries = ms.sweep_entries()
    for name, entry in entries.items():
        assert entry["path"].name.startswith(name), entry["path"]
        assert entry["max_tries"] == 3
    assert list(entries) == sorted(entries, key=ms.size_key)


def test_the_whole_sweep_is_analysed():
    """Qwen3.5-122B-A10B sat in EXCLUDED while its run was in flight; it has
    since finished, so all twelve are scored."""
    assert ms.EXCLUDED == ()
    assert len(ms.sweep_entries()) == 12
    assert "Qwen3.5-122B-A10B" in ms.sweep_entries()


def test_a_named_exclusion_is_honoured():
    entries = ms.sweep_entries(exclude=("Qwen3.5-122B-A10B",))
    assert len(entries) == 11
    assert "Qwen3.5-122B-A10B" not in entries


def test_a_partial_run_is_refused_rather_than_scored_on_what_it_reached(monkeypatch):
    """The real protection, independent of EXCLUDED: a run still in flight
    merges cleanly and would silently shrink the complete-case intersection for
    every other model. Forgetting to list it must not be enough to include it.
    """
    human = _human(n_participants=10)
    short = _model(human, human["new_belief"].to_numpy()).head(5)

    monkeypatch.setattr(ms, "load_human_normalized_data", lambda: human)
    monkeypatch.setattr(ms, "load_normalized_data", lambda path: short)

    with pytest.raises(SystemExit, match="still in flight"):
        ms.load({"Partial-9B": CONFIGS_DIR / "model_size"})


# --------------------------------------------------------------------------
# the retry column
# --------------------------------------------------------------------------

def _workbook(tmp_path, tries, name="m.xlsx"):
    path = tmp_path / name
    pd.DataFrame({"persona_id": [f"P{i}" for i in range(len(tries))],
                  "n_tries": tries}).to_excel(path, index=False)
    return path


def test_retry_rate_counts_rows_that_used_every_allowed_attempt(tmp_path):
    path = _workbook(tmp_path, [1, 1, 2, 3, 3, 3])
    rates = ms.retry_rates({"M": {"path": path, "max_tries": 3}})
    assert rates["M"]["n"] == 3
    assert rates["M"]["total"] == 6
    assert rates["M"]["share"] == pytest.approx(0.5)


def test_the_retry_ceiling_comes_from_the_config_not_a_literal(tmp_path):
    """A run given 5 tries has not struggled by reaching 3, so the column is
    counted against that run's own budget."""
    path = _workbook(tmp_path, [1, 3, 3, 5])
    assert ms.retry_rates({"M": {"path": path, "max_tries": 5}})["M"]["n"] == 1
    assert ms.retry_rates({"M": {"path": path, "max_tries": 3}})["M"]["n"] == 2


def test_retries_are_counted_over_all_rows_not_the_intersection(tmp_path):
    """Unlike every score in the table. A model's retry rate describes its run,
    and dropping the rows other models failed would understate it."""
    path = _workbook(tmp_path, [3] * 10)
    assert ms.retry_rates({"M": {"path": path, "max_tries": 3}})["M"]["total"] == 10


# --------------------------------------------------------------------------
# p-values, on the complete-case frame
# --------------------------------------------------------------------------

def test_a_perfect_model_has_no_paired_difference():
    human = _human()
    y = human["new_belief"].to_numpy()
    frame, _ = mr.assemble(human, {"Perfect-1B": _model(human, y)})
    row = ms.permutation_rows(frame, ["Perfect-1B"], n_resamples=2000)["Perfect-1B"]
    assert row["delta"] == pytest.approx(0.0)
    assert not row["significant"]


def test_a_constant_offset_is_recovered_as_delta_and_is_significant():
    human = _human()
    y = human["new_belief"].to_numpy()
    frame, _ = mr.assemble(human, {"Off1-1B": _model(human, y + 1)})
    row = ms.permutation_rows(frame, ["Off1-1B"], n_resamples=5000)["Off1-1B"]
    assert row["delta"] == pytest.approx(-1.0)     # human - model
    assert row["p"] < 0.001
    assert row["significant"]


def test_p_values_are_holm_corrected_across_the_sweep():
    for row in _result()["perm"].values():
        assert row["p_holm"] >= row["p"]


def test_the_p_value_and_the_mae_describe_the_same_cases():
    """permutation_test_on_paths would re-merge from the files and test each
    model on its own rows; then a row's p would not belong with its MAE."""
    human = _human()
    y = human["new_belief"].to_numpy()
    models = {"A-1B": _model(human, y + 1), "B-2B": _model(human, y)}
    frame, _ = mr.assemble(human, models)
    rows = ms.permutation_rows(frame, list(models), n_resamples=2000)
    r = mr.compute(frame, list(models), n_boot=100, n_perm=200)
    for name in models:
        assert rows[name]["delta"] == pytest.approx(
            r["metrics"]["bias"][name]["estimate"])


# --------------------------------------------------------------------------
# p formatting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p,expected", [
    (0.0, "< 0.0001"),
    (1e-7, "< 0.0001"),
    (0.00009, "< 0.0001"),
    (0.0001, "0.0001"),
    (0.02076, "0.0208"),
    (1.0, "1.0000"),
])
def test_p_values_round_to_four_places_with_a_floor(p, expected):
    assert ms.p_text(p) == expected


def test_a_p_value_is_never_printed_as_zero():
    """0.0000 would claim the null was excluded exactly, which no resampling
    test can do -- and fast_permutation_test really does return 0.0, since its
    resolution is 1/1e6."""
    assert "0.0000" not in ms.p_text(0.0)


# --------------------------------------------------------------------------
# the CSV handover
# --------------------------------------------------------------------------

def test_the_csv_carries_everything_the_renderer_needs():
    table = _table()
    for column in ("source", "family", "params_total_b", "params_active_b",
                   "is_baseline", "is_sparse", "valid_share", "tries_share",
                   "max_tries", "mae", "mae_lo", "mae_hi", "rank",
                   "delta", "delta_lo", "delta_hi",
                   "p", "p_holm", "significant", "significant_raw"):
        assert column in table.columns, column


def test_the_two_intervals_are_for_different_quantities():
    """MAE and Delta are different statistics; a single CI cannot serve both.
    Delta's brackets zero where the test does not reject, MAE's never can."""
    table = _table()
    assert not table["delta_lo"].equals(table["mae_lo"])
    for _, row in table.iterrows():
        assert row["delta_lo"] <= row["delta"] <= row["delta_hi"]
        assert row["mae_lo"] <= row["mae"] <= row["mae_hi"]
        assert row["mae_lo"] >= 0


def test_the_delta_interval_agrees_with_the_permutation_test():
    """Two independent procedures on the same statistic -- a bootstrap and a
    sign-flip test. If a CI excluded zero while p said 0.9, one of them is
    wired to the wrong column."""
    for _, row in _table().iterrows():
        excludes_zero = row["delta_lo"] > 0 or row["delta_hi"] < 0
        if excludes_zero:
            assert row["p"] < 0.10, row["source"]


def test_run_metadata_rides_on_every_row_so_the_csv_stands_alone():
    """The caption is written from this file. Splitting it across two files is
    how a table ends up describing a different run than its numbers."""
    table = _table()
    for column in ("n_obs", "n_participants", "n_boot", "seed", "alpha", "excluded"):
        assert table[column].nunique(dropna=False) == 1, column


def test_baselines_are_present_but_carry_no_size_or_rank():
    table = _table()
    baselines = table[table["is_baseline"]]
    assert not baselines.empty
    assert baselines["params_active_b"].isna().all()
    assert baselines["rank"].isna().all()


def test_the_csv_survives_a_round_trip(tmp_path):
    """Everything the renderer reads has to come back out of CSV text."""
    path = tmp_path / "ms.csv"
    _table().to_csv(path, index=False)
    assert mst.render(pd.read_csv(path)).startswith("%")


# --------------------------------------------------------------------------
# rendering, from the CSV alone
# --------------------------------------------------------------------------

def _body(tex):
    """The data rows: not the header, not the family group headers."""
    after_head = tex.split(r"\midrule", 1)[1]
    return [l for l in after_head.splitlines()
            if l.endswith(r"\\") and not l.startswith(r"\multicolumn")]


def _first_cell(line):
    return line.split("&")[0].strip().replace(r"\textit{", "").rstrip("}")


def _body_names(tex):
    """Rendered labels in order, skipping stacked continuation rows (blank)."""
    return [n for n in map(_first_cell, _body(tex)) if n]


def _label(row, family_headers=False):
    """The label as the table renders it: the family prefix is dropped only
    when a family header carries it."""
    if family_headers:
        return mst.short_label(row["source"], row["family"])
    return mst.display_name(row["source"])


def _models(table):
    return table[~table["is_baseline"].astype(bool)]


def _line_index(body, row, family_headers=False):
    label = _label(row, family_headers)
    for j, line in enumerate(body):
        if _first_cell(line) == label:
            return j
    raise AssertionError(f"no row labelled {label!r} in:\n" + "\n".join(body))


def _line(body, row, family_headers=False):
    return body[_line_index(body, row, family_headers)]


def test_latex_rows_have_as_many_cells_as_the_header():
    tex = mst.render(_table())
    assert {l.count("&") for l in _body(tex)} == {7}
    header = next(l for l in tex.splitlines() if l.startswith("Model &"))
    assert header.count("&") == 7


@pytest.mark.parametrize("source,family,expected", [
    ("gemma-4-26B-A4B-it", "gemma-4", "26B-A4B-it"),
    ("Qwen3.5-0.8B", "Qwen3.5", "0.8B"),
    ("Qwen3.5-122B-A10B", "Qwen3.5", "122B-A10B"),
    ("Standalone-9B", "gemma-4", "Standalone-9B"),   # no prefix to strip
])
def test_the_family_prefix_is_dropped_from_the_row_label(source, family, expected):
    """The group header above the row already says the family. Repeating it made
    the first column the widest in the table, which is part of why $p$ fell off
    the page."""
    assert mst.short_label(source, family) == expected


def test_every_model_is_still_identifiable_from_its_group():
    """Stripping must not collide two models onto one label."""
    table = _table()
    models = table[~table["is_baseline"]]
    labels = [(r["family"], mst.short_label(r["source"], r["family"]))
              for _, r in models.iterrows()]
    assert len(set(labels)) == len(models)


def test_families_stay_contiguous_without_a_heading_row():
    """The grouping is still there -- a family's models sit together, separated
    by \\addlinespace -- but no heading row announces it."""
    tex = mst.render(_table())
    assert r"\multicolumn" not in tex
    assert r"\addlinespace" in tex

    table = _table()
    by_label = {mst.display_name(r["source"]): r["family"]
                for _, r in _models(table).iterrows()}
    families = [by_label[n] for n in _body_names(tex) if n in by_label]
    assert len(families) == len(by_label)
    assert families == sorted(families, key=families.index), families
    assert len(set(families)) == 2


def test_labels_keep_the_family_prefix_when_no_heading_carries_it():
    """Without a heading row, a bare '1B' would not say whose 1B it is."""
    assert "Acme-1B" in _body_names(mst.render(_table()))


def test_family_headings_can_be_restored_and_then_shorten_the_labels():
    tex = mst.render(_table(), family_headers=True)
    assert r"\multicolumn{8}{l}{\textit{Acme}}" in tex
    names = _body_names(tex)
    assert "1B" in names and "Acme-1B" not in names


def test_within_a_family_rows_are_ordered_by_active_parameters():
    """Acme-30B-A2B activates 2B, so it sits between the 1B and the 8B rather
    than after both on its 30B of stored weights."""
    assert _body_names(mst.render(_table()))[:3] == [
        "Acme-1B", "Acme-30B-A2B", "Acme-8B"]


def test_sorting_by_total_moves_the_sparse_model_and_nothing_else():
    """The flag has to actually change something, or --sort is decoration."""
    assert _body_names(mst.render(_table(), sort="total"))[:3] == [
        "Acme-1B", "Acme-8B", "Acme-30B-A2B"]


def test_a_sparse_model_shows_both_its_active_and_stored_size():
    table = _table()
    row = table[table["source"] == "Acme-30B-A2B"].iloc[0]
    assert "2 / 30" in _line(_body(mst.render(table)), row)


def test_a_dense_model_shows_one_size():
    table = _table()
    row = table[table["source"] == "Acme-1B"].iloc[0]
    assert "/" not in _line(_body(mst.render(table)), row).split("&")[1]


def _with_p(*values):
    """The fixture with its four models' p-values replaced, in row order.

    The fixture's own p-values are all near 0 or 1, so a test about where the
    Bonferroni line falls has to place values on either side of it.
    """
    table = _table()
    idx = _models(table).index
    table.loc[idx, "p"] = list(values)
    table.loc[idx, "p_holm"] = 1.0       # must not influence anything rendered
    return table, [table.loc[i] for i in idx]


def _p_cell(tex, row):
    return _line(_body(tex), row).split("&")[-1].replace(r"\\", "").strip()


def test_the_p_column_prints_the_raw_uncorrected_p():
    """Bonferroni decides only the bold. The number printed is the test's own
    p, never p times the number of tests -- nor the Holm value in the CSV."""
    table, rows = _with_p(0.0200, 0.5, 0.5, 0.5)
    cell = _p_cell(mst.render(table), rows[0])
    assert "0.0200" in cell
    assert "0.0800" not in cell and "1.0000" not in cell


def test_bold_marks_p_below_the_bonferroni_threshold_over_the_models():
    """Four models at alpha 0.05 put the line at 0.0125: 0.0100 is bold,
    0.0200 is significant uncorrected but not after correction, so plain."""
    table, rows = _with_p(0.0100, 0.0200, 0.0124, 0.9)
    tex = mst.render(table)
    assert _p_cell(tex, rows[0]) == r"$\mathbf{0.0100}$"
    assert _p_cell(tex, rows[1]) == "$0.0200$"
    assert _p_cell(tex, rows[2]) == r"$\mathbf{0.0124}$"
    assert _p_cell(tex, rows[3]) == "$0.9000$"


def test_the_bonferroni_count_ignores_the_baselines_in_the_csv():
    """Baselines are in the CSV but not the table; dividing by six instead of
    four would put the line at 0.0083 and un-bold this 0.0100."""
    assert len(_table()) > len(_models(_table()))
    table, rows = _with_p(0.0100, 0.5, 0.5, 0.5)
    assert r"\mathbf" in _p_cell(mst.render(table), rows[0])


def test_the_bold_compares_the_unrounded_p():
    """0.01249 prints as 0.0125 but is below 0.0125; 0.012501 prints the same
    and is not. The threshold applies to the value, not the printed digits."""
    table, rows = _with_p(0.01249, 0.012501, 0.5, 0.5)
    tex = mst.render(table)
    assert r"\mathbf" in _p_cell(tex, rows[0])
    assert r"\mathbf" not in _p_cell(tex, rows[1])


def test_a_floored_p_is_bold_inside_one_math_group():
    table, rows = _with_p(1e-7, 0.5, 0.5, 0.5)
    assert _p_cell(mst.render(table), rows[0]) == r"$\mathbf{< 0.0001}$"


def test_alpha_moves_the_bonferroni_line():
    table, rows = _with_p(0.0200, 0.5, 0.5, 0.5)
    assert r"\mathbf" in _p_cell(mst.render(table, alpha=0.10), rows[0])


def test_the_caption_states_the_correction_and_that_p_is_raw():
    caption = mst.render(_table()).split(r"\caption{")[1].split("\n")[0]
    assert "uncorrected" in caption
    assert "Bonferroni correction across the 4 models" in caption
    assert "$p < 0.05/4$" in caption
    assert "Holm" not in caption


def test_baselines_are_not_rendered():
    tex = mst.render(_table())
    assert "No change" not in tex and "Cell median" not in tex
    assert "reference predictors" not in tex
    assert r"\textit{" not in "\n".join(_body(tex))


def test_families_are_separated_by_a_rule():
    """A spaced midrule between families; the only other midrule is under the
    header, so two in total for the fixture's two families."""
    tex = mst.render(_table())
    assert "\\addlinespace\n\\midrule\n\\addlinespace" in tex
    assert tex.count(r"\midrule") == 2


@pytest.mark.parametrize("name,expected", [
    ("gemma-4-E2B-it", "Gemma-4-E2B-it"),
    ("Qwen3.5-27B", "Qwen3.5-27B"),
    ("", ""),
])
def test_display_names_start_with_a_capital(name, expected):
    assert mst.display_name(name) == expected


def test_the_caption_matches_the_trimmed_wording():
    caption = mst.render(_table()).split(r"\caption{")[1].split("\n")[0]
    assert "what the models are ranked on" in caption
    assert "cancel opposing errors" not in caption
    assert "served checkpoints" not in caption
    assert "Whole participants are resampled" not in caption
    assert r"\emph" not in caption


def _interval(cell):
    """The two numbers out of a '[lo, hi]' cell, whatever wraps it."""
    inner = cell.split("[")[1].split("]")[0]
    return tuple(float(v) for v in inner.split(","))


@pytest.mark.parametrize("ci", ["stacked", "inline"])
def test_delta_carries_its_own_interval(ci):
    """The permutation test is on the paired SIGNED difference, so the interval
    beside it has to be the one for that quantity -- the MAE CI is an interval
    for something else and would not even contain zero when p says it should."""
    table = _table()
    row = table[~table["is_baseline"]].sort_values("delta").iloc[0]
    body = _body(mst.render(table, ci=ci))
    i = _line_index(body, row)
    cells = (body[i] if ci == "inline" else body[i + 1]).split("&")

    assert _interval(cells[-2]) == (pytest.approx(row["delta_lo"]),
                                    pytest.approx(row["delta_hi"]))
    assert _interval(cells[4]) == (pytest.approx(row["mae_lo"]),
                                   pytest.approx(row["mae_hi"]))


def test_stacked_is_the_default_and_puts_each_interval_under_its_estimate():
    """Inline, the two interval columns alone overrun a one-column page and
    push the p column off the edge -- which is how this was noticed."""
    table = _table()
    body = _body(mst.render(table))
    assert len(body) == 2 * len(_models(table)), \
        "expected one estimate and one CI row each"

    row = table[~table["is_baseline"]].iloc[0]
    i = _line_index(body, row)
    assert "[" not in body[i], "the estimate row must not carry the interval"
    assert body[i + 1].split("&")[0].strip() == "", "the CI row repeats no label"
    assert r"\footnotesize" in body[i + 1]


def test_inline_keeps_one_row_per_model():
    table = _table()
    assert len(_body(mst.render(table, ci="inline"))) == len(_models(table))


def test_the_header_names_the_interval_only_when_it_is_inline():
    table = _table()
    assert r"MAE [95\% CI]" in mst.render(table, ci="inline")
    assert r"MAE [95\% CI]" not in mst.render(table)


@pytest.mark.parametrize("font,expected", [
    ("small", r"\small"), ("footnotesize", r"\footnotesize"),
])
def test_the_body_font_is_selectable_and_reset_afterwards(font, expected):
    tex = mst.render(_table(), font=font)
    assert expected + "\n" in tex
    assert tex.index(expected + "\n") < tex.index(r"\begin{tabular}")
    assert r"\normalsize" in tex


def test_normal_font_adds_no_size_command():
    tex = mst.render(_table(), font="normal")
    assert r"\normalsize" not in tex


def test_an_empty_exclusion_list_does_not_become_the_word_nan(tmp_path):
    """EXCLUDED = () writes an empty CSV field, pandas reads it back as NaN, and
    NaN is truthy -- so `x or ''` kept it and str() made it "nan". That reached
    the caption as "Excluded, run unfinished: nan"."""
    path = tmp_path / "ms.csv"
    table = _table()
    table["excluded"] = ""
    table.to_csv(path, index=False)
    tex = mst.render(pd.read_csv(path))
    assert "nan" not in tex
    assert "Excluded" not in tex


def test_a_real_exclusion_is_still_reported(tmp_path):
    path = tmp_path / "ms.csv"
    table = _table()
    table["excluded"] = "Zeta-99B"
    table.to_csv(path, index=False)
    assert "Excluded, run unfinished: Zeta-99B." in mst.render(pd.read_csv(path))


def test_latex_keeps_each_p_cell_in_one_math_group():
    """`$0.02$$^{*}$` renders, but two adjacent groups are one whitespace edit
    away from opening display math."""
    assert "$$" not in mst.render(_table())


def test_latex_escapes_percent_signs_in_the_share_columns():
    """An unescaped % comments out the rest of the line, silently eating the
    MAE, the rank and the p-value."""
    for line in mst.render(_table()).splitlines():
        if line.endswith(r"\\"):
            assert "%" not in line.replace(r"\%", "")


def test_the_caption_reports_the_run_that_produced_the_numbers():
    table = _table()
    tex = mst.render(table)
    assert str(int(table.iloc[0]["n_obs"])) in tex
    assert str(int(table.iloc[0]["n_boot"])) in tex


def test_a_split_retry_budget_is_refused_rather_than_mislabelled():
    """One column cannot say 'used every retry' for runs allowed different
    numbers of them."""
    table = _table()
    table.loc[table.index[0], "max_tries"] = 5
    with pytest.raises(ValueError, match="different retry budgets"):
        mst.render(table)


def test_a_stale_csv_is_refused_rather_than_half_rendered():
    table = _table().drop(columns=["delta_lo"])
    with pytest.raises(ValueError, match="regenerate it"):
        mst.render(table)


def test_the_text_report_carries_the_size_columns_and_the_caveat():
    text = ms.render(_result())
    assert "active" in text
    assert "valid%" in text
    assert "tries%" in text
    assert "not a random subset" in text
