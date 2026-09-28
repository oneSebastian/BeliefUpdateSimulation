"""
Higher-order belief distributions for all 6 models, broken down by topic: each
model's `general_public_stance` (bars) against the participants'
`second_order_belief` (outline). Same layout as post_stance_by_topic_models.

Both questions ask about the statement as it was worded for that participant
("What do you belief the average stance of the public is on this statement?"),
so both are flipped into the pro-topic frame like the first-order beliefs.

Run from: repository root
Output:   figures/plots/distributions/general_public_stance_by_topic_models.svg / .png / .pdf
"""

from belief_update_sim.data_loading import load_normalized_data, load_human_normalized_data
from belief_update_sim.normalization import flip_negative
from scripts.figures.distributions.post_stance_by_topic_models import (
    MODELS, plot_grid, split_by_topic,
)


def load_model_gps_by_topic(path):
    # load_normalized_data already flips general_public_stance.
    return split_by_topic(load_normalized_data(path), "general_public_stance")


def normalize_human_gps(df):
    """Flip participants' second-order beliefs into the pro-topic frame."""
    return flip_negative(df, ["second_order_belief"])


def load_human_gps_by_topic():
    df = normalize_human_gps(load_human_normalized_data())
    return split_by_topic(df, "second_order_belief")


def main():
    print("Loading data...")
    model_data = {label: load_model_gps_by_topic(path) for label, path in MODELS}
    plot_grid(model_data, load_human_gps_by_topic(),
              "figures/plots/distributions/general_public_stance_by_topic_models")
    print("Done.")


if __name__ == "__main__":
    main()
