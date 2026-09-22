#!/bin/bash
#
# Submit model runs, sizing each SLURM allocation from its config.
#
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json             # one full run
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json --pilot     # one smoke test
#     ./slurm/submit_model.sh --pilot-all                            # pilot every model
#     ./slurm/submit_model.sh --all --resume                         # full sweep
#
# --gpus, --time and --partition are fixed when sbatch is called, not when the
# job runs, so they are read here and passed on the command line, where they
# override the #SBATCH defaults inside run_model.slurm.
#
# BATCH MODES ARE SEQUENTIAL BY DEFAULT. Each job is submitted with
# --dependency=afterany on the one before it, so at most one model is resident
# at a time and the rest of the node stays free for other users. `afterany`
# rather than `afterok`: a model that OOMs or fails to load must not strand the
# eleven behind it.
#
#     --parallel        submit batch jobs without the dependency chain; SLURM
#                       will then run as many as fit in the node's GPUs at once
#     --after JOBID     chain the first job behind an existing job, e.g. to
#                       queue the full sweep behind the pilots
#     --only GLOB       restrict a batch to configs whose name matches, for
#                       re-running one family after a fix:
#                           ./slurm/submit_model.sh --pilot-all --only 'gemma-4-*'
#
# Pilot jobs request the config's `serving.pilot_time` rather than its full-run
# walltime -- 15 calls do not need 48 hours, and asking for them would hold a
# slot the job cannot use. pilot_time is sized per model to cover a FIRST run,
# where the weights still have to be downloaded: 2h for Qwen3.5-0.8B (~2GB) up
# to 12h for 122B-A10B (~244GB), against a pessimistic ~10 MB/s.
#
# Set PILOT_TIME_LIMIT to override every model at once on a slower link:
#     PILOT_TIME_LIMIT=24:00:00 ./slurm/submit_model.sh --pilot-all
#
# Pre-fetching instead keeps the reservations small:
#     hf download Qwen/Qwen3.5-122B-A10B
#
# Arguments that are not recognised here are forwarded verbatim to run_agent.py.

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BATCH_MODE=""          # "pilot" | "full" | ""
SEQUENTIAL=1
AFTER=""
ONLY=""                # glob over config basenames, for re-running a subset
PASSTHROUGH=()

while [ $# -gt 0 ]; do
    case "$1" in
        --pilot-all) BATCH_MODE="pilot"; shift ;;
        --all)       BATCH_MODE="full";  shift ;;
        --parallel)  SEQUENTIAL=0;       shift ;;
        --only)
            if [ $# -lt 2 ]; then
                echo "ERROR: --only needs a glob, e.g. --only 'gemma-4-*'." >&2
                exit 2
            fi
            ONLY="$2"; shift 2 ;;
        --after)
            if [ $# -lt 2 ]; then
                echo "ERROR: --after needs a job id." >&2
                exit 2
            fi
            AFTER="$2"; shift 2 ;;
        --sequential) SEQUENTIAL=1; shift ;;   # explicit form of the default
        *)           PASSTHROUGH+=("$1"); shift ;;
    esac
done

usage() {
    echo "usage: $0 <config> [run_agent args...]" >&2
    echo "       $0 --pilot-all [--only GLOB] [--parallel] [--after JOBID]" >&2
    echo "       $0 --all [--resume] [--only GLOB] [--parallel] [--after JOBID]" >&2
    echo "" >&2
    echo "configs:" >&2
    ls configs/model_size/*.json 2>/dev/null | sed 's|^|  |' >&2
}

if [ -z "$BATCH_MODE" ] && [ ${#PASSTHROUGH[@]} -lt 1 ]; then
    usage
    exit 2
fi

# ---------------------------------------------------------------------------
# Environment
#
# SKIP_CONDA=1 for an already-activated shell (and for the tests, which shim
# sbatch rather than submitting anything).
# ---------------------------------------------------------------------------
if [ "${SKIP_CONDA:-0}" != "1" ]; then
    CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    CONDA_ENV="${CONDA_ENV:-vllm}"

    if [ ! -f "$CONDA_SH" ]; then
        echo "ERROR: conda profile not found at '$CONDA_SH'." >&2
        echo "       Set CONDA_SH, or SKIP_CONDA=1 if the env is already active." >&2
        exit 1
    fi
    # `conda activate` trips the nounset guard in some conda versions.
    set +u
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    set -u
fi

mkdir -p logs

PYTHON="${PYTHON:-python}"
PREV_JOBID="$AFTER"

# ---------------------------------------------------------------------------
# Submit one config
# ---------------------------------------------------------------------------
submit_one() {
    local config="$1"; shift

    # Validates as a side effect: a bad config fails here, not in the queue.
    local params
    params="$("$PYTHON" -m scripts.pipeline.serving_params "$config" --format sh)"
    eval "$params"

    local suffix=""
    for arg in "$@"; do
        [ "$arg" = "--pilot" ] && suffix="-pilot"
    done

    # A pilot is 15 calls. Reserving the full-run walltime for it would hold a
    # multi-hour slot the job cannot use and lose backfill priority. The config's
    # own pilot_time is sized per model to cover a first-run weight download
    # (2h for Qwen3.5-0.8B, 12h for 122B-A10B); PILOT_TIME_LIMIT overrides all
    # of them at once if the link is slower than that.
    if [ -n "$suffix" ]; then
        TIME_LIMIT="${PILOT_TIME_LIMIT:-$PILOT_TIME}"
    fi

    local dep=()
    if [ "$SEQUENTIAL" -eq 1 ] && [ -n "$PREV_JOBID" ]; then
        dep=(--dependency="afterany:$PREV_JOBID")
    fi

    local jobid
    jobid="$(sbatch --parsable \
        "${dep[@]}" \
        --job-name="${JOB_NAME}${suffix}" \
        --gpus="$GPUS" \
        --time="$TIME_LIMIT" \
        --partition="$PARTITION" \
        slurm/run_model.slurm "$config" "$@")"
    # On a federated cluster --parsable appends ";clustername".
    jobid="${jobid%%;*}"

    local after_note=""
    if [ ${#dep[@]} -gt 0 ]; then
        after_note=" (after $PREV_JOBID)"
    fi
    echo "  job $jobid  ${MODEL}${suffix}  ${GPUS} GPU(s)  ${TIME_LIMIT}  ${PARTITION}${after_note}"

    PREV_JOBID="$jobid"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [ -n "$BATCH_MODE" ]; then
    configs=()
    for candidate in configs/model_size/*.json; do
        [ -e "$candidate" ] || continue
        if [ -n "$ONLY" ]; then
            # shellcheck disable=SC2254 -- $ONLY is a glob on purpose
            case "$(basename "$candidate" .json)" in
                $ONLY) ;;
                *) continue ;;
            esac
        fi
        configs+=("$candidate")
    done

    if [ ${#configs[@]} -eq 0 ]; then
        if [ -n "$ONLY" ]; then
            echo "ERROR: no configs in configs/model_size/ match '$ONLY'." >&2
        else
            echo "ERROR: no configs found in configs/model_size/." >&2
        fi
        exit 1
    fi

    # Cheapest allocation first. The 4-GPU job can sit pending while other users
    # hold cards; submitting it last means the eleven ahead of it have already
    # produced results by the time it waits, instead of stalling the chain at
    # the front. Within a GPU tier, roughly smallest model first.
    # tr -d '\r': a Python writing text-mode newlines (Windows) would otherwise
    # leave a carriage return that mapfile -t does not strip, turning every
    # path into one that does not exist.
    ordered="$("$PYTHON" -m scripts.pipeline.serving_params --sort-by-gpus "${configs[@]}" | tr -d '\r')" || {
        echo "ERROR: could not order configs by GPU count." >&2
        exit 1
    }
    mapfile -t configs <<< "$ordered"

    extra=()
    [ "$BATCH_MODE" = "pilot" ] && extra=(--pilot)

    if [ "$SEQUENTIAL" -eq 1 ]; then
        echo "Submitting ${#configs[@]} ${BATCH_MODE} job(s), chained -- one model resident at a time."
    else
        echo "Submitting ${#configs[@]} ${BATCH_MODE} job(s) unchained -- SLURM may run several at once."
    fi
    echo "Ordered by GPU count, cheapest first."
    [ -n "$AFTER" ] && echo "First job waits for job $AFTER."

    for config in "${configs[@]}"; do
        submit_one "$config" ${extra[@]+"${extra[@]}"} ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}
    done

    echo ""
    echo "Last job in the chain: $PREV_JOBID"
    echo "  watch:  squeue -u \$USER"
    if [ "$BATCH_MODE" = "pilot" ]; then
        echo "  then:   $PYTHON -m scripts.pipeline.pilot_report 'results/model_size/pilot/*.xlsx'"
        echo "  queue the full sweep behind these:"
        echo "          $0 --all --after $PREV_JOBID"
    fi
    exit 0
fi

submit_one "${PASSTHROUGH[@]}"
