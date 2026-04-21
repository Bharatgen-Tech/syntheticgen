#!/bin/bash
# Resume test: run summary pipeline twice on no-id seeds (hash-based ids).
# First run generates outputs with sha1_* seed_ids.
# Second run should skip both seeds in <30s total (no GPU work).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/test_output_resume"
rm -rf "${OUTPUT_DIR}"      # fresh start
mkdir -p "${OUTPUT_DIR}"

export PATH="$HOME/miniconda3/bin:$PATH"
export HF_HOME="/opt/dlami/nvme"
export HF_HUB_CACHE="/opt/dlami/nvme/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN() {
  python3 -m synthgen.cli run synthgen/pipelines/summary.yaml \
    --model "Qwen/Qwen3-Coder-Next" \
    --input "${SCRIPT_DIR}/test_seeds_no_id.jsonl" \
    --output "${OUTPUT_DIR}/out.jsonl" \
    --intermediate "${OUTPUT_DIR}/debug.jsonl" \
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

cd "${SCRIPT_DIR}"

echo "=== RUN 1: fresh start (expect 2 seeds processed) ==="
RUN
echo ""
echo "=== Output after run 1: ==="
cat "${OUTPUT_DIR}/out.jsonl"
echo ""

echo "=== RUN 2: same input, should skip both (resume) ==="
T0=$(date +%s)
RUN
T1=$(date +%s)
echo ""
echo "=== Run 2 took $((T1 - T0))s ==="
echo "(Expected <60s since no generation should happen; engine still warms up briefly)"
echo ""
echo "=== Output after run 2 (should be identical): ==="
cat "${OUTPUT_DIR}/out.jsonl"
