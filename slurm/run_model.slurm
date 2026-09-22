#!/bin/bash
#SBATCH --job-name=belief-sim
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=verylong
#
# Generic runner: serves ONE model with vLLM and runs the agent against it.
# Every model-specific parameter -- huggingface id, GPUs, tensor-parallel size,
# context length, walltime -- comes from the config's `serving` block, so this
# script is the same for all twelve models in the size sweep.
#
# Submit it through the wrapper, which reads --gpus/--time/--partition out of
# the same config (they are fixed at submit time, not run time, so the #SBATCH
# defaults above are only a fallback):
#
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json            # full run
#     ./slurm/submit_model.sh model_size/Qwen3.5-9B.json --pilot    # smoke test
#
# Arguments after the config are forwarded verbatim to run_agent.py.

set -uo pipefail

CONFIG="${1:-}"
if [ -z "$CONFIG" ]; then
    echo "usage: sbatch slurm/run_model.slurm <config> [run_agent args...]" >&2
    echo "       (normally invoked through slurm/submit_model.sh)" >&2
    exit 2
fi
shift
AGENT_ARGS=("$@")


# ---------------------------------------------------------------------------
# 0. Activate conda environment
#
# Site-specific. Override by exporting these before submitting:
#     CONDA_SH=/path/to/conda/etc/profile.d/conda.sh CONDA_ENV=myenv sbatch ...
# ---------------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-vllm}"

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda profile not found at '$CONDA_SH'." >&2
    echo "       Set CONDA_SH to your cluster's conda profile.d/conda.sh." >&2
    exit 1
fi
source "$CONDA_SH"

if ! conda activate "$CONDA_ENV"; then
    echo "ERROR: could not activate conda env '$CONDA_ENV'." >&2
    exit 1
fi

mkdir -p logs


# ---------------------------------------------------------------------------
# 1. Read the serving parameters out of the config
#
# serving_params validates before it prints and writes errors to stderr, so a
# malformed config fails here rather than after a 40-minute model load.
# ---------------------------------------------------------------------------
PARAMS="$(python -m scripts.pipeline.serving_params "$CONFIG" --format sh)" || {
    echo "ERROR: could not read serving parameters from '$CONFIG'." >&2
    exit 1
}
eval "$PARAMS"

echo "[$(date)] config           : $CONFIG"
echo "[$(date)] model            : $MODEL ($HF_MODEL_ID)"
echo "[$(date)] gpus             : $GPUS (tensor-parallel $TENSOR_PARALLEL_SIZE)"
echo "[$(date)] context/completion: $MAX_MODEL_LEN / $MAX_TOKENS tokens"
echo "[$(date)] port             : $PORT"
echo "[$(date)] agent args       : ${AGENT_ARGS[*]:-(none)}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv || true


# ---------------------------------------------------------------------------
# 2. Start the vLLM server
# ---------------------------------------------------------------------------
# Nothing here may JIT-compile a CUDA kernel. The system toolkit at
# /usr/bin/nvcc is too old for the flags FlashInfer emits -- it rejects
# `--compress-mode=size` outright -- and there is no newer one on the cluster.
# Every such build dies inside the profile run, after the weights have already
# loaded, which costs ten minutes before the job fails.
#
# The sampler: the native/Triton path needs no nvcc and draws from the same
# distribution, so results are unaffected. Set here rather than per model, so
# every model in the sweep samples through the same kernel -- otherwise the
# kernel would be a confound on the size axis.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# The all-reduce: this one only bites at tensor-parallel > 1, which is why the
# first pilot wave lost exactly the five multi-GPU models and kept the seven
# single-GPU ones. vLLM 0.30 promoted FLASHINFER to the front of the all-reduce
# dispatch order; 0.27.1 left it in the candidate list but chose CUSTOM. Once
# dispatch goes through FlashInfer, the fused all-reduce+RMSNorm JIT-compiles
# trtllm_mnnvl_allreduce.cu, and that is the build that fails.
#
# Turning it off restores what 0.27.1 did (dispatch falls back to CUSTOM, then
# PYNCCL). Note this is NOT the same as disabling the fusion pass: a working
# 0.27.1 run on 4 GPUs had `fuse_allreduce_rms: True` and never called nvcc,
# because the fused op only builds that kernel when it dispatches to FlashInfer.
# Leave the pass alone; it is the backend choice that regressed.
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"

VLLM_LOG="logs/vllm_${MODEL}_${SLURM_JOB_ID:-local}.out"
echo "[$(date)] Starting vllm server -> $VLLM_LOG"
echo "[$(date)] VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
echo "[$(date)] VLLM_ALLREDUCE_USE_FLASHINFER=$VLLM_ALLREDUCE_USE_FLASHINFER"
echo "[$(date)] nvcc: $(command -v nvcc || echo '(none on PATH)')"
echo "[$(date)] vllm serve ${VLLM_ARGS[*]}"

vllm serve "${VLLM_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "[$(date)] vllm server PID: $VLLM_PID"

shutdown_vllm() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date)] Shutting down vllm server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null
        # Give it 60s to release the GPUs before forcing it.
        for _ in $(seq 12); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 5
        done
        kill -9 "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null
    fi
}
# Covers scancel and walltime expiry too, not just the happy path.
trap 'shutdown_vllm; exit 143' TERM INT


# ---------------------------------------------------------------------------
# 3. Wait until the server is ready
#
# Large checkpoints take a while to load (122B-A10B is ~244GB of weights), so
# the timeout is generous -- but a server that has already died is detected
# immediately rather than waited out.
# ---------------------------------------------------------------------------
MAX_WAIT="${VLLM_STARTUP_TIMEOUT:-2400}"
INTERVAL=10
ELAPSED=0

echo "[$(date)] Waiting up to ${MAX_WAIT}s for vllm to become ready..."
until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date)] ERROR: vllm exited during startup. Last 50 lines:" >&2
        tail -n 50 "$VLLM_LOG" >&2
        exit 1
    fi
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        echo "[$(date)] ERROR: vllm not ready within ${MAX_WAIT}s. Last 50 lines:" >&2
        tail -n 50 "$VLLM_LOG" >&2
        shutdown_vllm
        exit 1
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "[$(date)] vllm server is ready (waited ${ELAPSED}s)."


# ---------------------------------------------------------------------------
# 4. Run the agent
# ---------------------------------------------------------------------------
AGENT_LOG="logs/agent_${MODEL}_${SLURM_JOB_ID:-local}.out"
echo "[$(date)] Running agent -> $AGENT_LOG"

python -m scripts.pipeline.run_agent --config "$CONFIG" "${AGENT_ARGS[@]}" \
    > "$AGENT_LOG" 2>&1
AGENT_EXIT=$?
echo "[$(date)] agent exited with code $AGENT_EXIT"
tail -n 20 "$AGENT_LOG"


# ---------------------------------------------------------------------------
# 5. On a pilot, report on the token budget and format compliance
# ---------------------------------------------------------------------------
PILOT=0
for arg in ${AGENT_ARGS[@]+"${AGENT_ARGS[@]}"}; do
    [ "$arg" = "--pilot" ] && PILOT=1
done

REPORT_EXIT=0
if [ "$PILOT" -eq 1 ] && [ "$AGENT_EXIT" -eq 0 ]; then
    echo ""
    echo "[$(date)] ===== PILOT REPORT: $MODEL ====="
    python -m scripts.pipeline.pilot_report "results/model_size/pilot/${MODEL}_pilot.xlsx"
    REPORT_EXIT=$?
    if [ "$REPORT_EXIT" -ne 0 ]; then
        echo "[$(date)] Pilot flagged an issue -- adjust the config before the full run."
    fi
fi


# ---------------------------------------------------------------------------
# 6. Shut the server down
# ---------------------------------------------------------------------------
trap - TERM INT
shutdown_vllm
echo "[$(date)] Done."

if [ "$AGENT_EXIT" -ne 0 ]; then
    exit "$AGENT_EXIT"
fi
exit "$REPORT_EXIT"
