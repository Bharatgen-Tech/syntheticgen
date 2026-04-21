#!/bin/bash
# LCG smoke test via synthgen: 1 seed through the full think-execute pipeline.
# Equivalent to `bash run_ray_native.sh --limit 1` in the original repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LCG_DIR="/fsx/pretraining_data/nauman/github/long_context_generation"
OUTPUT_DIR="${SCRIPT_DIR}/test_output_lcg"
mkdir -p "${OUTPUT_DIR}"

export PATH="$HOME/miniconda3/bin:$PATH"
export HF_HOME="/opt/dlami/nvme"
export HF_HUB_CACHE="/opt/dlami/nvme/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LIMIT="${1:-20}"   # override: bash test_lcg.sh 50

echo "=== LCG smoke test via synthgen ==="
echo "Pipeline: long_context.yaml (think_execute / mode2)"
echo "Seeds:    ${LIMIT} (from codechef_final.jsonl)"
echo "GPUs:     8 (2 replicas, TP=4 each)"
echo ""

cd "${SCRIPT_DIR}"

python3 -m synthgen.cli run synthgen/pipelines/long_context.yaml \
  --model "Qwen/Qwen3-Coder-Next" \
  --input "${LCG_DIR}/codechef_final.jsonl" \
  --output "${OUTPUT_DIR}/out.jsonl" \
  --intermediate "${OUTPUT_DIR}/debug.jsonl" \
  --limit "${LIMIT}" \
  --num_replicas 2 \
  --tensor_parallel_size 4 \
  --gpus_per_replica 4 \
  --gpu_memory_utilization 0.92 \
  --max_num_seqs 512 \
  --max_workers 256 \
  --max_seed_concurrency 128 \
  --context_window 131072 \
  --enforce_eager \
  --enable_expert_parallel \
  --cuda_visible_devices "0,1,2,3,4,5,6,7"

echo ""
echo "=== Output record ==="
python3 -c "
import json
with open('${OUTPUT_DIR}/out.jsonl') as f:
    for line in f:
        r = json.loads(line)
        print(f'seed_id: {r[\"seed_id\"]}')
        print(f'task:    {r[\"task\"][:100]}...')
        print(f'words:   {r[\"word_count\"]:,}')
        print(f'preview: {r[\"text\"][:400]}...')
        print()
"
