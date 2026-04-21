# synthgen

Model-agnostic, pipeline-agnostic synthetic data generation on Ray + vLLM.

Define your pipeline in YAML, point at a model and a seed file, and scale from 1 GPU to N nodes with the same command.

## Layout

```
synthgen/
  engine.py            RayNativeEngine: multi-replica vLLM, NodeAffinity, fault tolerance, probe-gated startup
  writer.py            AsyncBufferedWriter: batched JSONL
  runner.py            Ray init, multi-node wait, seed loading, resume, orchestration
  pipeline.py          Pipeline + StageSpec + YAML loader + template renderer
  utils.py             Shared: token budgeting, context truncation, JSON repair, retry wrapper
  cli.py               `synthgen run pipeline.yaml ...`
  flows/
    linear.py          Sequential stage executor (with fanout support)
    think_execute.py   Pipelined think-execute overlap (port of async_lc.py mode2) — TODO
  pipelines/
    summary.yaml       Minimal single-stage smoke test
```

## Quick start — single node

```bash
python -m synthgen.cli run synthgen/pipelines/summary.yaml \
  --model Qwen/Qwen3-Coder-Next \
  --input seeds.jsonl \
  --output out/corpus.jsonl \
  --intermediate out/debug.jsonl \
  --num_replicas 2 \
  --tensor_parallel_size 4 \
  --gpus_per_replica 4 \
  --gpu_memory_utilization 0.92 \
  --max_workers 256 \
  --max_seed_concurrency 128 \
  --enforce_eager
```

## Multi-node (N nodes)

Start Ray on head + workers, then:

```bash
python -m synthgen.cli run synthgen/pipelines/summary.yaml \
  --model Qwen/Qwen3-Coder-Next \
  --input seeds.jsonl \
  --output out/corpus.jsonl \
  --intermediate out/debug.jsonl \
  --ray_address auto \
  --num_nodes 4 \
  --replicas_per_node 2 \
  --tensor_parallel_size 4 \
  --gpus_per_replica 4 \
  --gpu_memory_utilization 0.92 \
  --max_workers 512 \
  --max_seed_concurrency 256 \
  --max_num_seqs 512 \
  --enable_expert_parallel \
  --enforce_eager
```

## Pipeline YAML

```yaml
name: my_pipeline
flow: linear            # or 'think_execute'

stages:
  - name: draft
    system_prompt: "You are a helpful writer."
    prompt: |
      Seed: {seed.text}
      Write a draft.
    decoding: { temperature: 0.3, max_tokens: 4096 }

  - name: critique
    inputs: [draft]     # documentation; template vars work either way
    prompt: |
      Draft: {draft}
      List 3 weaknesses.
    decoding: { temperature: 0.3, max_tokens: 1024, parse: json }

  - name: revisions
    fanout: critique    # run once per critique item
    prompt: |
      Issue: {item}
      Propose a fix.
    decoding: { max_tokens: 512 }

merge:
  template: "{draft}\n\n# Critique\n{critique}\n\n# Revisions\n{revisions}"
```

New use case = new YAML. No code changes.

## Status

Phase 1 done (engine/writer/runner/pipeline/linear). `think_execute` flow still needs porting from `long_context_generation/async_lc.py`.
