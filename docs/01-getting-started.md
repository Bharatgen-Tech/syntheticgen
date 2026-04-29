# Getting started

You'll install synthgen, run a single-stage pipeline on two example seeds, and see output in 2 minutes.

## 1. Install

```bash
git clone https://github.com/Bharatgen-Tech/syntheticgen.git
cd syntheticgen
pip install -e .
```

Requirements:
- Python 3.10+
- At least one GPU
- vLLM-compatible CUDA drivers (`nvidia-smi` works)
- Model weights — by default we use `Qwen/Qwen3-Coder-Next`. The first run will either download them or use what's already in your HuggingFace cache.

If `pip install -e .` fails with `Disk quota exceeded` (common on shared Lustre filesystems), use a local temp dir:
```bash
TMPDIR=/tmp pip install -e .
```

## 2. Verify the install

```bash
synthgen --help
synthgen validate examples/qa_generation/pipeline.yaml
```

The second command loads the example pipeline YAML, checks every field, and prints any warnings. It costs nothing (no GPU, no model) and catches typos before you waste time on a real run.

## 3. Run your first pipeline

```bash
bash examples/qa_generation/run.sh
```

This pipeline:
1. Reads 3 seed documents from [`examples/qa_generation/seeds.jsonl`](../examples/qa_generation/seeds.jsonl)
2. For each, extracts 3 key concepts
3. For each concept, generates a question/answer pair

Output lands in `examples/qa_generation/out/qa.jsonl` — one record per input seed, containing the concepts list and QA pairs.

First run takes ~2 minutes: 1-2 minutes for the model to load into GPU memory, a few seconds for the actual generation. On reruns with warm cache, it's faster.

## 4. Inspect the output

```bash
cat examples/qa_generation/out/qa.jsonl | head -1 | python -m json.tool
```

You'll see a JSON record like:
```json
{
  "seed_id": "doc_001",
  "text": "Concepts: [...]\n\nQA pairs:\n[{...}, {...}, {...}]",
  "word_count": 234
}
```

## 5. Rerun — resume in action

Run the exact same command again:
```bash
bash examples/qa_generation/run.sh
```

Synthgen detects the 3 already-processed seeds and exits in under 5 seconds without loading the model:
```
[INFO] Resuming — 3 seed(s) already complete.
[INFO] 0 seed(s) to process.
[INFO] Nothing to do — skipping engine setup. Done.
```

To force a rerun, delete the output file: `rm examples/qa_generation/out/qa.jsonl`.

## 6. What's next

- **Run it on your own data** — replace `seeds.jsonl` with your inputs. One JSON record per line, must have a `text` field. An `id` or `filename` field is optional; if absent, synthgen assigns a stable sha1-based id.
- **Build a different pipeline** — see [02-build-a-pipeline.md](02-build-a-pipeline.md) for common recipes.
- **Scale across nodes** — see [03-scale-up.md](03-scale-up.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `nvidia-smi` hangs or fails | Your NVIDIA driver is dead. Reboot or pick another machine. |
| `Disk quota exceeded` during `pip install` | `TMPDIR=/tmp pip install -e .` |
| `PermissionError on .locks/` | Your HuggingFace cache lock dir is owned by another user. Either `sudo chmod -R a+rwX /path/to/.locks` or set `HF_HUB_OFFLINE=1` to skip locking (the example run.sh does this). |
| "GPUs busy" | Another process is using GPU memory. Run `nvidia-smi` to see what — leave it alone if it's someone else's work, `pkill -9 -f vllm` if it's a stale run of yours. |
