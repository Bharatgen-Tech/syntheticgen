#!/bin/bash
# Smoke test: run synthgen's summary pipeline on 2 seeds, 1 replica, 4 GPUs.
# Exercises: CLI parse -> YAML load -> Ray init -> RayNativeEngine deploy ->
#            probe warmup -> linear flow -> stage render -> engine.generate ->
#            buffered writer -> resume detection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/test_output"
mkdir -p "${OUTPUT_DIR}"

export PATH="$HOME/miniconda3/bin:$PATH"

# Use local NVMe cache (on ip-10-0-235-109) instead of NFS/Lustre.
# OFFLINE=1 skips the lock-file write that fails due to .locks dir perms.
export HF_HOME="/opt/dlami/nvme"
export HF_HUB_CACHE="/opt/dlami/nvme/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Synthgen smoke test ==="
echo "HF_HOME=${HF_HOME} (offline mode)"
echo "Pipeline: summary.yaml"
echo "Seeds: 2 (test_seeds.jsonl)"
echo "GPUs: 4 (1 replica, TP=4)"
echo ""

cd "${SCRIPT_DIR}"

python3 -m synthgen.cli run synthgen/pipelines/summary.yaml \
  --model "Qwen/Qwen3-Coder-Next" \
  --input "${SCRIPT_DIR}/test_seeds.jsonl" \
  --output "${OUTPUT_DIR}/out.jsonl" \
  --intermediate "${OUTPUT_DIR}/debug.jsonl" \
  --num_replicas 1 \
  --tensor_parallel_size 4 \
  --gpus_per_replica 4 \
  --gpu_memory_utilization 0.92 \
  --max_num_seqs 256 \
  --max_workers 16 \
  --max_seed_concurrency 4 \
  --context_window 8192 \
  --enforce_eager \
  --cuda_visible_devices "0,1,2,3"

echo ""
echo "=== Output ==="
cat "${OUTPUT_DIR}/out.jsonl"
echo ""
echo "=== Debug ==="
cat "${OUTPUT_DIR}/debug.jsonl"
