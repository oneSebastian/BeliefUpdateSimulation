from pathlib import Path

import pandas as pd

from .config import MERGED_HUMAN_CSV, PROJECT_ROOT
from .normalization import flip_negative


def resolve(path):
    """Resolve a possibly repo-relative path against the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_data(df):
    return flip_negative(df, ["llm_new_belief", "llm_general_public_stance"])


def load_normalized_data(path):
    df = pd.read_excel(resolve(path))
    df = normalize_data(df)
    df = df.rename(columns={
        "llm_new_belief": "new_belief",
        "llm_general_public_stance": "general_public_stance",
        "llm_reasoning": "reasoning",
        "init_belief": "init_stance",
    })
    return df


def load_human_normalized_data():
    df = pd.read_csv(MERGED_HUMAN_CSV)
    df = df[['persona_id', 'topic', 'topic_position', 'package', 'message_order_code', 'message_1_shown',
       'message_2_shown', 'message_3_shown', 'statement_formulation',
       'initial_belief', 'final_belief', 'familiarity', 'second_order_belief',
       'stance_message_1_shown', 'stance_message_2_shown',
       'stance_message_3_shown', 'rank_message_1_shown',
       'rank_message_2_shown', 'rank_message_3_shown', 'text_response', 'age',
       'gender', 'ethnicity', 'country_of_birth', 'country_of_residency',
       'student_status', 'employment_status', 'occupation_field',
       'highest_qualification', 'BIG_item_1', 'BIG_item_2', 'BIG_item_3',
       'BIG_item_4', 'BIG_item_5', 'BIG_item_6', 'BIG_item_7', 'BIG_item_8',
       'BIG_item_9', 'BIG_item_10', 'BIG_item_1_label', 'BIG_item_2_label',
       'BIG_item_3_label', 'BIG_item_4_label', 'BIG_item_5_label',
       'BIG_item_6_label', 'BIG_item_7_label', 'BIG_item_8_label',
       'BIG_item_9_label', 'BIG_item_10_label', 'BIG5_extraversion_score',
       'BIG5_agreeableness_score', 'BIG5_conscientiousness_score',
       'BIG5_neuroticism_score', 'BIG5_openness_score', 'initial_belief_normalized', 'final_belief_normalized']]
    df = df.rename(columns={
        'initial_belief': 'initial_belief_unnormalized',
        'final_belief': 'final_belief_unnormalized',
        'initial_belief_normalized': 'init_stance',
        'final_belief_normalized': 'new_belief',
    })
    return df
