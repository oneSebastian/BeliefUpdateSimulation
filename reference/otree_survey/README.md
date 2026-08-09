# oTree survey — reference copy

This directory is a static snapshot of the [oTree](https://www.otree.org/) survey
application that ran the human belief-update experiment. It is the exact
instrument the human participants interacted with: consent, initial-stance
elicitation, the message pages, post-stance elicitation, comment ranking, the
Big-5 questionnaire, and the closing pages.

## Purpose — reference only

It is included for **documentation and reporting**, so the survey design can be
inspected alongside the analysis. It is **not part of the analysis pipeline**:

- No code in this repository imports, runs, or depends on it.
- It lives outside the Python package and outside `tests/`, so it is excluded
  from packaging and from test collection.
- It is provided for reading, not execution. Running it would require a working
  oTree installation per this directory's own `requirements.txt`, which is the
  survey app's dependency list and is separate from the project's top-level
  `requirements.txt`.

## Relation to the analysed data

This application produced the raw oTree exports used in the analysis (for
example `data/400_participant_otree.xlsx`). The page names referenced in the
participant-flow accounting — `Consent`, `InitBeliefOne`, `MessagePageOne`,
`EndBeliefOne`, `RankingOne`, … `Big5Page`, `ThankYou` — are the page classes
defined here in `survey/`.

## License

Covered by this project's MIT `LICENSE`; no separate license applies.
