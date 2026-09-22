#!/bin/bash
#
# Submit one model's run, sizing the SLURM allocation from its config.
#
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json             # full run
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json --pilot     # smoke test
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json --resume    # restart
#
# --gpus, --time and --partition are fixed when sbatch is called, not when the
# job runs, so they are read here and passed on the command line, where they
# override the #SBATCH defaults inside run_model.slurm.
#
# With --pilot-all in place of a config, every model_size config is submitted as
# a pilot, each as its own job:
#
#     ./slurm/submit_model.sh --pilot-all

set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vllm}"

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda profile not found at '$CONDA_SH'." >&2
    exit 1
fi
# `conda activate` trips the nounset guard in some conda versions.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

mkdir -p logs

submit_one() {
    local config="$1"; shift

    # Validates as a side effect: a bad config fails here, not in the queue.
    local params
    params="$(python -m scripts.pipeline.serving_params "$config" --format sh)"
    eval "$params"

    local suffix=""
    for arg in "$@"; do
        [ "$arg" = "--pilot" ] && suffix="-pilot"
    done

    echo "submitting ${MODEL}${suffix}: ${GPUS} GPU(s), ${TIME_LIMIT} on ${PARTITION}"
    sbatch \
        --job-name="${JOB_NAME}${suffix}" \
        --gpus="$GPUS" \
        --time="$TIME_LIMIT" \
        --partition="$PARTITION" \
        slurm/run_model.slurm "$config" "$@"
}

if [ "${1:-}" = "--pilot-all" ]; then
    shift
    for config in configs/model_size/*.json; do
        submit_one "$config" --pilot "$@"
    done
    echo ""
    echo "All pilots submitted. Once they finish, review them together with:"
    echo "    python -m scripts.pipeline.pilot_report 'results/model_size/pilot/*.xlsx'"
    exit 0
fi

if [ $# -lt 1 ]; then
    echo "usage: $0 <config> [run_agent args...]" >&2
    echo "       $0 --pilot-all" >&2
    echo "" >&2
    echo "configs:" >&2
    ls configs/model_size/*.json 2>/dev/null | sed 's|^|  |' >&2
    exit 2
fi

submit_one "$@"
