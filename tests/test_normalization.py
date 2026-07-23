import pytest

from belief_update_sim.normalization import (
    NEGATIVE_FORMULATIONS,
    POSITIVE_FORMULATIONS,
    UnknownFormulationError,
    normalize_value,
    statement_polarity,
)

POS = "Everyone should receive Universal Basic Income."
NEG = "There should be no Universal Basic Income."


def test_positive_formulations_have_polarity_one():
    for f in POSITIVE_FORMULATIONS:
        assert statement_polarity(f) == 1


def test_negative_formulations_have_polarity_minus_one():
    for f in NEGATIVE_FORMULATIONS:
        assert statement_polarity(f) == -1


def test_every_topic_has_both_framings():
    assert len(POSITIVE_FORMULATIONS) == len(NEGATIVE_FORMULATIONS) == 3
    assert not (POSITIVE_FORMULATIONS & NEGATIVE_FORMULATIONS)


def test_unknown_formulation_is_fatal():
    """Must raise, not default to +1 -- a silent default would flip half a
    topic's rows into the wrong frame."""
    with pytest.raises(UnknownFormulationError):
        statement_polarity("Some statement we have never seen.")


def test_normalize_leaves_positive_framing_untouched():
    for v in (-2, -1, 0, 1, 2):
        assert normalize_value(v, POS) == v


def test_normalize_flips_negative_framing():
    for v in (-2, -1, 0, 1, 2):
        assert normalize_value(v, NEG) == -v


def test_normalize_is_its_own_inverse():
    for f in (POS, NEG):
        for v in (-2, -1, 0, 1, 2):
            assert normalize_value(normalize_value(v, f), f) == v


def test_normalize_passes_through_missing_values():
    assert normalize_value(None, NEG) is None
    assert normalize_value(float("nan"), NEG) is None
