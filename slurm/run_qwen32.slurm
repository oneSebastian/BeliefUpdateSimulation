#!/bin/bash
#SBATCH --job-name=qwen32-agent
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --partition=verylong


# ---------------------------------------------------------------------------
# 0. Activate conda environment
#
# Site-specific. Override for your cluster either by editing the defaults or by
# exporting these before submitting:
#     CONDA_SH=/path/to/conda/etc/profile.d/conda.sh CONDA_ENV=myenv sbatch ...
# ---------------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-belief-update-sim}"

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
PORT=8004

echo "[$(date)] Starting vllm server on port $PORT..."
vllm serve Qwen/Qwen3-32B \
    --served-model-name Qwen3-32B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --enable-auto-tool-choice \
    --tool-call-parser llama3_json \
    --port $PORT \
    > logs/vllm_qwen32_${SLURM_JOB_ID}.out 2>&1 &

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
# 3. Run the two agent scripts in parallel
# ---------------------------------------------------------------------------
echo "[$(date)] Launching agent scripts..."

CONFIG_T0="Qwen3-32B-t0.0.json"
python -m scripts.pipeline.run_agent --config ${CONFIG_T0} --resume > logs/agent_${CONFIG_T0%.json}_$$.out 2>&1 &
PID_T0=$!
# rename now that we have the PID
mv logs/agent_${CONFIG_T0%.json}_$$.out logs/agent_${CONFIG_T0%.json}_${PID_T0}.out

CONFIG_T2="Qwen3-32B-t2.0.json"
python -m scripts.pipeline.run_agent --config ${CONFIG_T2} --resume > logs/agent_${CONFIG_T2%.json}_$$.out 2>&1 &
PID_T2=$!
mv logs/agent_${CONFIG_T2%.json}_$$.out logs/agent_${CONFIG_T2%.json}_${PID_T2}.out

echo "[$(date)] agent ${CONFIG_T0} PID: $PID_T0 -> logs/agent_${CONFIG_T0%.json}_${PID_T0}.out"
echo "[$(date)] agent ${CONFIG_T2} PID: $PID_T2 -> logs/agent_${CONFIG_T2%.json}_${PID_T2}.out"

# ---------------------------------------------------------------------------
# 4. Wait for both scripts to finish, capture exit codes
# ---------------------------------------------------------------------------
wait $PID_T0; EXIT_T0=$?
wait $PID_T2; EXIT_T2=$?

echo "[$(date)] agent ${CONFIG_T0} exited with code $EXIT_T0"
echo "[$(date)] agent ${CONFIG_T2} exited with code $EXIT_T2"

# ---------------------------------------------------------------------------
# 5. Shut down the vllm server
# ---------------------------------------------------------------------------
echo "[$(date)] Shutting down vllm server (PID $VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "[$(date)] Done."

# Propagate failure if either agent script failed
[ $EXIT_T0 -ne 0 ] || [ $EXIT_T2 -ne 0 ] && exit 1
exit 0
