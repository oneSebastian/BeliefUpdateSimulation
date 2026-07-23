"""Single source of truth for belief-polarity normalization.

Each topic was presented under one of two statement formulations: one worded in
favour of the topic and one worded against it. A belief recorded against a
negatively-worded statement sits in the opposite frame from one recorded against
a positively-worded statement, so the sign must be flipped before values are
compared or pooled across formulations.

After normalization, positive always means agreement with the pro-topic framing
(pro-UBI, pro-penalty-shootouts, pro-weight-loss-drugs).

Ranks are never normalized: they index which message was shown, not a polarity.

NOTE: the figure scripts each carry their own copy of this flip, in four
slightly different variants (see git history / the scripts under figures/).
They are consistent for the columns each one actually uses, but new code should
import from here rather than adding a fifth copy.
"""

POSITIVE_FORMULATIONS = frozenset({
    "Everyone should receive Universal Basic Income.",
    "Penalty shootouts are a good way to determine the winners of football matches.",
    "Using weight loss drugs like Ozempic is a good way to lose weight.",
})

NEGATIVE_FORMULATIONS = frozenset({
    "There should be no Universal Basic Income.",
    "Penalty shootouts are a bad way to determine the winners of football matches.",
    "Using weight loss drugs like Ozempic is a bad way to lose weight.",
})

KNOWN_FORMULATIONS = POSITIVE_FORMULATIONS | NEGATIVE_FORMULATIONS


class UnknownFormulationError(ValueError):
    """Raised when a statement formulation has no known polarity.

    Deliberately fatal: silently treating an unrecognised formulation as
    positive would flip half a topic's data into the wrong frame.
    """


def statement_polarity(formulation):
    """Return +1 for pro-topic wording, -1 for anti-topic wording."""
    if formulation in POSITIVE_FORMULATIONS:
        return 1
    if formulation in NEGATIVE_FORMULATIONS:
        return -1
    raise UnknownFormulationError(f"unknown statement formulation: {formulation!r}")


def normalize_value(value, formulation):
    """Flip `value` into the pro-topic frame. None/NaN passes through."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value * statement_polarity(formulation)


def flip_negative(df, columns, formulation_col="statement_formulation"):
    """Flip the sign of `columns` on rows with a negative statement formulation.

    In place, returns df. This is the single implementation behind what used to
    be a copy-pasted `normalize()` in every figure script. Callers pass the
    exact columns they need flipped -- the scripts deliberately differ there
    (some also flip init_belief, some flip general_public_stance), so the column
    list stays with the caller while the masking logic lives here.
    """
    mask = df[formulation_col].isin(NEGATIVE_FORMULATIONS)
    df.loc[mask, columns] *= -1
    return df
