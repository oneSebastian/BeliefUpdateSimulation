"""Project paths, resolved from this file's location.

Replaces the previous implicit contract that every script be run with the
current working directory set to the repository root. Import these constants
instead of writing root-relative string literals, and scripts work regardless
of where they are invoked from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Inputs
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPT_TEMPLATES_DIR = PROJECT_ROOT / "prompt_templates"
CONFIGS_DIR = PROJECT_ROOT / "configs"
COLORS_YAML = PROJECT_ROOT / "colors.yaml"

MERGED_HUMAN_CSV = RESULTS_DIR / "merged_llm_participants_data_with_normalized_beliefs.csv"
CLEANED_HUMAN_CSV = DATA_DIR / "cleaned_for_llm_391_participant_otree.csv"
PROLIFIC_DATA_DIR = DATA_DIR / "prolific_data"
HF_DATASET_DIR = DATA_DIR / "hf_dataset"

# Generated artifacts -- kept out of the source tree.
# Figures are the exception: they are written to figures/plots/ so that the
# published images sit next to the panels assembled from them.
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATS_OUTPUT_DIR = OUTPUTS_DIR / "stats"
# Subset runs (one topic, or one topic x package cell) write here, so that the
# whole-design results in STATS_OUTPUT_DIR are never overwritten by them.
GROUPED_STATS_DIR = STATS_OUTPUT_DIR / "grouped"
# Regenerable intermediates (was figures/derived/): per-model regression tables,
# bias summaries, the combined-results CSV the zscore figures read, and the
# ablation tables written by personas.py / post_training.py.
DERIVED_DIR = OUTPUTS_DIR / "derived"


def resolve_config_path(config):
    """Resolve a config given either as a path or as a name under ``configs/``.

    Both forms occur, and callers should not have to know which they hold:

        gpt-5.2.json                         -> configs/gpt-5.2.json
        model_size/Qwen3.5-9B.json           -> configs/model_size/Qwen3.5-9B.json
        configs/model_size/Qwen3.5-9B.json   -> itself

    The last form is what ``slurm/submit_model.sh`` produces, because it builds
    its list by globbing ``configs/model_size/*.json``. ``--config`` has always
    meant the first two. Resolving both here is what stops the two conventions
    from disagreeing -- they previously did, and the result was a request for
    ``configs/configs/model_size/...`` that only failed after a model had been
    loaded on the cluster.
    """
    path = Path(config)
    if path.is_absolute() or path.exists():
        return path
    return CONFIGS_DIR / config


def ensure_output_dirs():
    """Create the output directories if they do not yet exist."""
    for path in (STATS_OUTPUT_DIR, DERIVED_DIR):
        path.mkdir(parents=True, exist_ok=True)
