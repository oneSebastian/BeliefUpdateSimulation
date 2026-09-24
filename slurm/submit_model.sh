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
# BATCH MODES ARE PACKED INTO WAVES. Jobs are greedily grouped so that each
# wave totals at most --max-gpus (default 4). A wave's jobs run concurrently;
# every job in a wave depends on ALL jobs of the wave before it. That keeps the
# node busy without ever asking for more than the node has -- and, crucially,
# the jobs that are waiting are held by a *dependency*, so SLURM does not have
# them requesting resources and they cannot take a backfill reservation away
# from another user. (A job pending on resources can; a job pending on
# Dependency cannot.)
#
# `afterany` rather than `afterok`: a model that OOMs or fails to load must not
# strand the waves behind it.
#
#     --max-gpus N      GPUs one wave may use (default 4)
#     --sequential      one job at a time; the same as --max-gpus 1
#     --parallel        no dependencies at all; SLURM decides everything, and
#                       queued jobs DO request resources
#     --after JOBID     hold the first wave behind an existing job, e.g. to
#                       queue the full sweep behind the pilots
#     --only GLOB       restrict a batch to configs whose name matches. Takes a
#                       comma-separated list, and the selection is still packed
#                       into waves as one batch:
#                           ./slurm/submit_model.sh --pilot-all --only 'gemma-4-*'
#                           ./slurm/submit_model.sh --all --only 'Qwen3.5-27B,gemma-4-31B-it'
#                       The list form is what re-running after a failure needs:
#                       the set left over is rarely one glob. There is no pattern
#                       that picks out the five tensor-parallel models without
#                       also catching gemma-4-12B-it.
#
# Pilot jobs request the config's `serving.pilot_time` rather than its full-run
# walltime -- 9 calls do not need 48 hours, and asking for them would hold a
# slot the job cannot use. pilot_time is sized per model to cover a FIRST run,
# where the weights still have to be downloaded.
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
MAX_GPUS=4
CHAINED=1              # 0 only for --parallel
AFTER=""
ONLY=""
PASSTHROUGH=()

while [ $# -gt 0 ]; do
    case "$1" in
        --pilot-all) BATCH_MODE="pilot"; shift ;;
        --all)       BATCH_MODE="full";  shift ;;
        --parallel)  CHAINED=0;          shift ;;
        --sequential) MAX_GPUS=1;        shift ;;
        --max-gpus)
            if [ $# -lt 2 ]; then
                echo "ERROR: --max-gpus needs a number." >&2
                exit 2
            fi
            MAX_GPUS="$2"; shift 2 ;;
        --after)
            if [ $# -lt 2 ]; then
                echo "ERROR: --after needs a job id." >&2
                exit 2
            fi
            AFTER="$2"; shift 2 ;;
        --only)
            if [ $# -lt 2 ]; then
                echo "ERROR: --only needs a glob or comma-separated list," >&2
                echo "       e.g. --only 'gemma-4-*' or --only 'Qwen3.5-27B,gemma-4-31B-it'." >&2
                exit 2
            fi
            ONLY="$2"; shift 2 ;;
        *)           PASSTHROUGH+=("$1"); shift ;;
    esac
done

usage() {
    echo "usage: $0 <config> [run_agent args...]" >&2
    echo "       $0 --pilot-all [--max-gpus N] [--only GLOB] [--after JOBID]" >&2
    echo "       $0 --all [--resume] [--max-gpus N] [--only GLOB] [--after JOBID]" >&2
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
LAST_JOBID=""

# ---------------------------------------------------------------------------
# Submit one config. $DEP_SPEC (may be empty) is the dependency to apply.
# ---------------------------------------------------------------------------
DEP_SPEC=""

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
    if [ -n "$suffix" ]; then
        TIME_LIMIT="${PILOT_TIME_LIMIT:-$PILOT_TIME}"
    fi

    local dep=()
    local note=""
    if [ -n "$DEP_SPEC" ]; then
        dep=(--dependency="afterany:$DEP_SPEC")
        note=" (after ${DEP_SPEC//:/, })"
    fi

    # Only the configs that declare `mem` get a --mem flag; the rest keep the
    # partition default, which has been enough for every model that fits on one
    # or two cards. Asking for memory a config did not ask for would make jobs
    # pend behind each other for no reason.
    local mem=()
    if [ -n "${MEM:-}" ]; then
        mem=(--mem="$MEM")
    fi

    local jobid
    jobid="$(sbatch --parsable \
        "${dep[@]}" \
        "${mem[@]}" \
        --job-name="${JOB_NAME}${suffix}" \
        --gpus="$GPUS" \
        --time="$TIME_LIMIT" \
        --partition="$PARTITION" \
        slurm/run_model.slurm "$config" "$@")"
    # On a federated cluster --parsable appends ";clustername".
    jobid="${jobid%%;*}"

    echo "    job $jobid  ${MODEL}${suffix}  ${GPUS} GPU(s)  ${TIME_LIMIT}  ${PARTITION}${MEM:+  ${MEM} RAM}${note}"
    LAST_JOBID="$jobid"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
if [ -n "$BATCH_MODE" ]; then
    configs=()
    for candidate in configs/model_size/*.json; do
        [ -e "$candidate" ] || continue
        if [ -n "$ONLY" ]; then
            name="$(basename "$candidate" .json)"
            matched=0
            # `set -f` so splitting $ONLY on commas cannot also glob it against
            # the working directory; the patterns are for the loop below only.
            saved_ifs="$IFS"
            set -f
            IFS=','
            for pattern in $ONLY; do
                # shellcheck disable=SC2254 -- $pattern is a glob on purpose
                case "$name" in
                    $pattern) matched=1; break ;;
                esac
            done
            IFS="$saved_ifs"
            set +f
            [ "$matched" -eq 1 ] || continue
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

    extra=()
    [ "$BATCH_MODE" = "pilot" ] && extra=(--pilot)

    # tr -d '\r': a Python writing text-mode newlines (Windows) would otherwise
    # leave a carriage return that mapfile -t does not strip.
    plan="$("$PYTHON" -m scripts.pipeline.serving_params \
                --plan-waves --max-gpus "$MAX_GPUS" "${configs[@]}" | tr -d '\r')" || {
        echo "ERROR: could not plan waves for the given configs." >&2
        exit 1
    }
    mapfile -t plan_lines <<< "$plan"

    if [ "$CHAINED" -eq 1 ]; then
        echo "Submitting ${#configs[@]} ${BATCH_MODE} job(s) in waves of at most ${MAX_GPUS} GPU(s)."
        echo "Waiting jobs are held by dependency, so they request no resources."
    else
        echo "Submitting ${#configs[@]} ${BATCH_MODE} job(s) unchained -- queued jobs WILL request resources."
    fi
    [ -n "$AFTER" ] && echo "First wave waits for job $AFTER."

    prev_wave_ids="$AFTER"     # colon-joined ids of the wave before this one
    this_wave_ids=""
    current_wave=""

    for line in "${plan_lines[@]}"; do
        [ -n "$line" ] || continue
        wave="${line%% *}"
        config="${line#* }"

        if [ "$wave" != "$current_wave" ]; then
            if [ -n "$current_wave" ]; then
                prev_wave_ids="$this_wave_ids"
                this_wave_ids=""
            fi
            current_wave="$wave"
            echo "  -- wave $wave --"
        fi

        if [ "$CHAINED" -eq 1 ]; then
            DEP_SPEC="$prev_wave_ids"
        else
            DEP_SPEC=""
        fi

        submit_one "$config" ${extra[@]+"${extra[@]}"} ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

        if [ -n "$this_wave_ids" ]; then
            this_wave_ids="${this_wave_ids}:${LAST_JOBID}"
        else
            this_wave_ids="$LAST_JOBID"
        fi
    done

    echo ""
    echo "Final wave: ${this_wave_ids//:/, }"
    echo "  watch:  squeue -u \$USER"
    if [ "$BATCH_MODE" = "pilot" ]; then
        echo "  then:   $PYTHON -m scripts.pipeline.pilot_report 'results/model_size/pilot/*.xlsx'"
        echo "  queue the full sweep behind these:"
        echo "          $0 --all --after ${this_wave_ids}"
    fi
    exit 0
fi

DEP_SPEC="$AFTER"
submit_one "${PASSTHROUGH[@]}"
