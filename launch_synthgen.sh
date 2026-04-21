#!/bin/bash
# Multi-node Ray + synthgen launcher. Role-based (head | worker).
#
# Usage:
#   On worker nodes (via SSH):  bash launch_synthgen.sh worker
#   On head node:               bash launch_synthgen.sh head [LIMIT]
#
# Assumes shared /fsx filesystem across all nodes.

set -euo pipefail

# ─── EDIT THESE ─────────────────────────────────────────────────
HEAD_IP="10.0.235.109"
WORKER_IPS=(
  "10.0.130.168"
  # add more here for 3+ nodes
)
# ────────────────────────────────────────────────────────────────

RAY_PORT=6379
NUM_NODES=$(( 1 + ${#WORKER_IPS[@]} ))
REPLICAS_PER_NODE=2
GPUS_PER_REPLICA=4
TENSOR_PARALLEL_SIZE=4

SCRIPT_DIR="/fsx/pretraining_data/nauman/github/synthgen"
LCG_DIR="/fsx/pretraining_data/nauman/github/long_context_generation"
PIPELINE="${SCRIPT_DIR}/synthgen/pipelines/long_context.yaml"
INPUT="${LCG_DIR}/codechef_final.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/test_output_multinode"
LIMIT="${2:-40}"      # default 40 seeds; override: bash launch_synthgen.sh head 100

export PATH="$HOME/miniconda3/bin:$HOME/.local/bin:$PATH"
export HF_HOME="/opt/dlami/nvme"
export HF_HUB_CACHE="/opt/dlami/nvme/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RAY="$HOME/miniconda3/bin/ray"
PY="$HOME/miniconda3/bin/python3"

ROLE="${1:-head}"

case "$ROLE" in
  head)
    echo "=== PURGING orphan Ray/vLLM processes on HEAD ==="
    $RAY stop --force 2>/dev/null || true
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'raylet' 2>/dev/null || true
    pkill -9 -f 'gcs_server' 2>/dev/null || true
    pkill -9 -f 'plasma_store' 2>/dev/null || true
    pkill -9 -f 'vllm' 2>/dev/null || true
    sleep 2

    echo "=== PURGING orphan Ray/vLLM processes on ${#WORKER_IPS[@]} worker(s) ==="
    for W in "${WORKER_IPS[@]}"; do
      ssh -o StrictHostKeyChecking=no "$W" \
        "\$HOME/miniconda3/bin/ray stop --force 2>/dev/null; \
         pkill -9 -f 'ray::'       2>/dev/null; \
         pkill -9 -f 'raylet'      2>/dev/null; \
         pkill -9 -f 'gcs_server'  2>/dev/null; \
         pkill -9 -f 'plasma_store' 2>/dev/null; \
         pkill -9 -f 'vllm'        2>/dev/null; \
         true" &
    done
    wait
    sleep 3

    echo "=== PRE-FLIGHT GPU CHECK (abort if any GPU is busy with someone else's work) ==="
    MAX_USED_MIB=5000    # >5 GiB used = someone else is using this GPU
    BUSY=0

    check_node() {
      local label="$1" host="$2"
      echo "[${label} ${host}]"
      local output
      if [ "${host}" = "local" ]; then
        output=$(nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits)
      else
        output=$(ssh -o StrictHostKeyChecking=no "$host" \
          "nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits")
      fi
      echo "${output}" | awk -v lbl="${label}" -v host="${host}" -v max="${MAX_USED_MIB}" '
        { gsub(/,/,""); used=$2;
          status = (used+0 > max ? "BUSY (used="used" MiB)" : "OK");
          printf "  GPU %s: used=%s MiB free=%s MiB  [%s]\n", $1, $2, $3, status;
          if (used+0 > max) exit 2
        }
      '
      return $?
    }

    if ! check_node "HEAD" "local"; then BUSY=1; fi
    for W in "${WORKER_IPS[@]}"; do
      if ! check_node "WORKER" "$W"; then BUSY=1; fi
    done

    if [ "$BUSY" -eq 1 ]; then
      echo ""
      echo "=== ABORT: one or more GPUs are in use by another process (>${MAX_USED_MIB} MiB used)."
      echo "=== Cannot proceed — these GPUs belong to someone else's workload."
      echo "=== Options:"
      echo "===   1. Pick different node(s) and edit WORKER_IPS"
      echo "===   2. Wait for the other user to finish"
      echo "===   3. Verify with 'ssh <node> nvidia-smi' who owns the processes"
      exit 1
    fi
    echo "=== All GPUs clean — proceeding ==="
    echo ""

    echo "=== STAGING model to workers' /opt/dlami/nvme (skips files already present) ==="
    MODEL_NAME="models--Qwen--Qwen3-Coder-Next"
    MODEL_SRC="/opt/dlami/nvme/hub/${MODEL_NAME}"
    if [ ! -d "${MODEL_SRC}" ]; then
      echo "!! Model not found at ${MODEL_SRC} on HEAD. Cannot stage to workers."
      exit 1
    fi

    for W in "${WORKER_IPS[@]}"; do
      echo "  -> ${W}"
      ssh -o StrictHostKeyChecking=no "$W" "mkdir -p /opt/dlami/nvme/hub"
      # -a preserves perms/times, --size-only skips files that match size (fast re-runs),
      # --partial resumes interrupted transfers
      rsync -a --partial --size-only --info=progress2 \
        "${MODEL_SRC}/" "${W}:/opt/dlami/nvme/hub/${MODEL_NAME}/" &
    done
    wait
    echo "=== Model staged on all workers ==="
    echo ""

    echo "=== STARTING Ray on HEAD ${HEAD_IP}:${RAY_PORT} ==="
    $RAY start --head --port="${RAY_PORT}" --num-gpus=8

    echo "=== STARTING Ray on ${#WORKER_IPS[@]} worker(s) via SSH ==="
    for W in "${WORKER_IPS[@]}"; do
      echo "  -> ${W}"
      ssh -o StrictHostKeyChecking=no "$W" \
        "export PATH=\$HOME/miniconda3/bin:\$PATH; \
         \$HOME/miniconda3/bin/ray start --address='${HEAD_IP}:${RAY_PORT}' --num-gpus=8" &
    done
    wait
    echo "=== All workers joined ==="

    sleep 5
    $RAY status || true

    mkdir -p "${OUTPUT_DIR}"
    cd "${SCRIPT_DIR}"

    echo ""
    echo "=== LAUNCHING synthgen on ${NUM_NODES} nodes × ${REPLICAS_PER_NODE} replicas/node = $((NUM_NODES * REPLICAS_PER_NODE)) replicas ==="
    echo "=== Limit: ${LIMIT} seeds ==="
    echo ""

    $PY -m synthgen.cli run "${PIPELINE}" \
      --model "Qwen/Qwen3-Coder-Next" \
      --input "${INPUT}" \
      --output "${OUTPUT_DIR}/out.jsonl" \
      --intermediate "${OUTPUT_DIR}/debug.jsonl" \
      --ray_address auto \
      --num_nodes "${NUM_NODES}" \
      --replicas_per_node "${REPLICAS_PER_NODE}" \
      --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
      --gpus_per_replica "${GPUS_PER_REPLICA}" \
      --gpu_memory_utilization 0.92 \
      --max_num_seqs 512 \
      --max_workers 512 \
      --max_seed_concurrency 256 \
      --context_window 131072 \
      --enforce_eager \
      --enable_expert_parallel \
      --limit "${LIMIT}"

    echo ""
    echo "=== Tearing down Ray on all nodes ==="
    $RAY stop --force 2>/dev/null || true
    for W in "${WORKER_IPS[@]}"; do
      ssh -o StrictHostKeyChecking=no "$W" "\$HOME/miniconda3/bin/ray stop --force" 2>/dev/null || true &
    done
    wait
    echo "=== Done. Output: ${OUTPUT_DIR}/out.jsonl ==="
    ;;

  worker)
    echo "=== WORKER: joining ${HEAD_IP}:${RAY_PORT} ==="
    $RAY stop --force 2>/dev/null || true
    $RAY start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=8
    echo "=== Worker joined. Run '$RAY status' to verify. ==="
    ;;

  *)
    echo "Usage: $0 {head|worker} [LIMIT]"
    exit 1
    ;;
esac
