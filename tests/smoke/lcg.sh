#!/bin/bash
# LCG smoke test: runs the long-context example through the think_execute flow.
# Default --limit 1. Override:  bash tests/smoke/lcg.sh 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out_lcg"
mkdir -p "${OUT_DIR}"

LIMIT="${1:-1}"

export PATH="$HOME/miniconda3/bin:$PATH"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/opt/dlami/nvme/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${REPO_ROOT}"

python3 -m synthgen.cli run examples/long_context/pipeline.yaml \
  --model "Qwen/Qwen3-Coder-Next" \
  --input examples/long_context/seeds.jsonl \
  --output "${OUT_DIR}/out.jsonl" \
  --intermediate "${OUT_DIR}/debug.jsonl" \
  --limit "${LIMIT}" \
  --num_replicas 2 \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.92 \
  --max_num_seqs 512 \
  --max_workers 256 \
  --max_seed_concurrency 128 \
  --context_window 131072 \
  --enforce_eager \
  --enable_expert_parallel \
  --cuda_visible_devices "0,1,2,3,4,5,6,7"

echo ""
echo "=== $(wc -l < "${OUT_DIR}/out.jsonl") record(s) in ${OUT_DIR}/out.jsonl ==="
