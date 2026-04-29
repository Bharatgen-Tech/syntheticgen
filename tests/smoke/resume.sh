#!/bin/bash
# Resume test: run summary pipeline twice on no-id seeds (hash-based ids).
# Run 1: generates outputs with sha1_* seed_ids.
# Run 2: should exit in <5s with "Nothing to do — skipping engine setup".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out_resume"
rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}"

export PATH="$HOME/miniconda3/bin:$PATH"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/opt/dlami/nvme/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${REPO_ROOT}"

RUN() {
  python3 -m synthgen.cli run synthgen/pipelines/summary.yaml \
    --model "Qwen/Qwen3-Coder-Next" \
    --input "${SCRIPT_DIR}/seeds_no_id.jsonl" \
    --output "${OUT_DIR}/out.jsonl" \
    --intermediate "${OUT_DIR}/debug.jsonl" \
    --num_replicas 1 \
    --tensor_parallel_size 4 \
    --gpus_per_replica 4 \
    --gpu_memory_utilization 0.92 \
    --max_num_seqs 128 \
    --max_workers 8 \
    --max_seed_concurrency 2 \
    --context_window 4096 \
    --enforce_eager \
    --cuda_visible_devices "0,1,2,3"
}

echo "=== RUN 1: fresh start ==="
RUN

echo ""
echo "=== RUN 2: same input, should skip both ==="
T0=$(date +%s)
RUN
T1=$(date +%s)
echo "=== Run 2 took $((T1 - T0))s (expect <5s with the zero-pending optimization) ==="
