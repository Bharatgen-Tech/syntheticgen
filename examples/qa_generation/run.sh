#!/bin/bash
# Q&A generation example — 3 seeds, 4 GPUs, 1 replica.
# Run from the synthgen repo root after `pip install -e .`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EX_DIR="${REPO_ROOT}/examples/qa_generation"
OUT_DIR="${EX_DIR}/out"
mkdir -p "${OUT_DIR}"

export PATH="$HOME/miniconda3/bin:$PATH"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${REPO_ROOT}"

python3 -m synthgen.cli run "${EX_DIR}/pipeline.yaml" \
  --model "Qwen/Qwen3-Coder-Next" \
  --input "${EX_DIR}/seeds.jsonl" \
  --output "${OUT_DIR}/qa.jsonl" \
  --intermediate "${OUT_DIR}/debug.jsonl" \
  --num_replicas 1 \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.92 \
  --max_num_seqs 128 \
  --max_workers 16 \
  --max_seed_concurrency 4 \
  --context_window 8192 \
  --enforce_eager \
  --cuda_visible_devices "0,1,2,3"

echo ""
echo "=== Output ==="
cat "${OUT_DIR}/qa.jsonl"
