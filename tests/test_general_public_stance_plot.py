"""Data handling behind the general-public-stance-by-topic figure.

The figure modules read colors.yaml relative to the working directory, so these
tests assume pytest is run from the repository root (as testpaths implies).
"""
import pandas as pd

from scripts.figures.distributions import general_public_stance_by_topic_models as gps
from scripts.figures.distributions.post_stance_by_topic_models import (
    MODELS, TOPICS, plot_grid, split_by_topic,
)

UBI_POS = "Everyone should receive Universal Basic Income."
UBI_NEG = "There should be no Universal Basic Income."
PEN_POS = "Penalty shootouts are a good way to determine the winners of football matches."
WL_NEG = "Using weight loss drugs like Ozempic is a bad way to lose weight."


def test_split_by_topic_groups_and_drops_missing():
    df = pd.DataFrame({
        "statement_formulation": [UBI_POS, UBI_NEG, PEN_POS, WL_NEG, WL_NEG],
        "value": [1, -2, 0, 2, None],
    })
    out = split_by_topic(df, "value")
    assert set(out) == set(TOPICS)
    assert sorted(out["UBI"]) == [-2, 1]
    assert list(out["Penalty"]) == [0]
    assert list(out["Weight Loss"]) == [2]
    assert out["Weight Loss"].dtype.kind == "i"


def test_human_second_order_belief_is_flipped_for_negative_wording():
    df = pd.DataFrame({
        "statement_formulation": [UBI_POS, UBI_NEG, WL_NEG],
        "second_order_belief": [1, 1, -2],
    })
    out = gps.normalize_human_gps(df)
    assert list(out["second_order_belief"]) == [1, -1, 2]


def test_human_loader_uses_normalized_second_order_belief(monkeypatch):
    df = pd.DataFrame({
        "statement_formulation": [UBI_POS, UBI_NEG, PEN_POS],
        "second_order_belief": [2, 2, -1],
        "new_belief": [0, 0, 0],  # must not be what gets plotted
    })
    monkeypatch.setattr(gps, "load_human_normalized_data", lambda: df.copy())
    out = gps.load_human_gps_by_topic()
    assert sorted(out["UBI"]) == [-2, 2]
    assert list(out["Penalty"]) == [-1]
    assert out["Weight Loss"].empty


def test_model_loader_reads_general_public_stance(monkeypatch):
    df = pd.DataFrame({
        "statement_formulation": [UBI_POS, PEN_POS],
        "general_public_stance": [1, -1],
        "new_belief": [2, 2],
    })
    monkeypatch.setattr(gps, "load_normalized_data", lambda path: df.copy())
    out = gps.load_model_gps_by_topic("unused.xlsx")
    assert list(out["UBI"]) == [1]
    assert list(out["Penalty"]) == [-1]


def test_plot_grid_writes_all_formats(tmp_path):
    values = pd.Series([-2, -1, 0, 1, 2, 0])
    data = {t: values for t in TOPICS}
    out = tmp_path / "grid"
    plot_grid({label: data for label, _ in MODELS}, data, str(out))
    for fmt in ("svg", "png", "pdf"):
        assert (tmp_path / f"grid.{fmt}").stat().st_size > 0
