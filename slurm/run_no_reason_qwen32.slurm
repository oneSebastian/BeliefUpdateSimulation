#!/bin/bash
#SBATCH --job-name=qwen32-no-reason
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --partition=verylong
#
# The `no-reason` ablation for Qwen3-32B: the main simulation, except that the
# model is not asked for a simulated reason for its belief update. Thinking
# stays on, as in the main run. Serving is identical to run_qwen32.slurm.
#
#     sbatch slurm/run_no_reason_qwen32.slurm            # full run (resumes)
#     sbatch slurm/run_no_reason_qwen32.slurm --pilot    # 3-persona smoke test
#
# Arguments are forwarded verbatim to run_agent.py.


# ---------------------------------------------------------------------------
# 0. Activate conda environment
#
# Site-specific. Override for your cluster either by editing the defaults or by
# exporting these before submitting:
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
    echo "       Set CONDA_ENV to an environment with requirements.txt installed." >&2
    exit 1
fi

# SBATCH --output/--error above need this to exist at submit time, not just here.
mkdir -p logs


# ---------------------------------------------------------------------------
# 1. Start the vllm server on the two GPUs Slurm allocated
# ---------------------------------------------------------------------------
PORT=8004   # must match "port" in configs/Qwen3-32B.json

echo "[$(date)] Starting vllm server on port $PORT..."
vllm serve Qwen/Qwen3-32B \
    --served-model-name Qwen3-32B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --port $PORT \
    > logs/vllm_qwen32_no-reason_${SLURM_JOB_ID}.out 2>&1 &

VLLM_PID=$!
echo "[$(date)] vllm server PID: $VLLM_PID"

# ---------------------------------------------------------------------------
# 2. Wait until the server is ready
# ---------------------------------------------------------------------------
echo "[$(date)] Waiting for vllm server to become ready..."
MAX_WAIT=900   # seconds
ELAPSED=0
INTERVAL=5

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "[$(date)] ERROR: vllm server did not start within ${MAX_WAIT}s. Aborting."
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "[$(date)] vllm server is ready (waited ${ELAPSED}s)."

# ---------------------------------------------------------------------------
# 3. Run the agent
# ---------------------------------------------------------------------------
CONFIG="Qwen3-32B.json"
AGENT_LOG="logs/agent_${CONFIG%.json}_no-reason_${SLURM_JOB_ID}.out"
echo "[$(date)] Running agent -> $AGENT_LOG"

python -m scripts.pipeline.run_agent --config ${CONFIG} --ablation no-reason --resume "$@" \
    > "$AGENT_LOG" 2>&1
AGENT_EXIT=$?

echo "[$(date)] agent exited with code $AGENT_EXIT"

# ---------------------------------------------------------------------------
# 4. Shut down the vllm server
# ---------------------------------------------------------------------------
echo "[$(date)] Shutting down vllm server (PID $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "[$(date)] Done."
exit $AGENT_EXIT
