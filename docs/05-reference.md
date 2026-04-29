# Reference

Lookup tables. Skim or search, don't read top-to-bottom.

## CLI

`synthgen` has two subcommands:

```bash
synthgen validate <pipeline.yaml>             # check a YAML for errors, no GPU
synthgen run <pipeline.yaml> [flags...]       # run the pipeline
```

### `run` flags

#### Required
| Flag | Description |
|---|---|
| `<pipeline.yaml>` (positional) | Path to the pipeline spec |
| `--model` | Model path or HuggingFace id (e.g. `Qwen/Qwen3-Coder-Next`) |
| `--input` | Input JSONL file |
| `--output` | Final output JSONL |
| `--intermediate` | Per-stage debug JSONL |

#### Multi-node
| Flag | Default | Description |
|---|---|---|
| `--ray_address` | — | `auto` to attach to an existing Ray cluster |
| `--num_nodes` | 1 | Expected node count (waits up to 10 min for all to join) |
| `--replicas_per_node` | 2 | vLLM replicas per node |

#### GPU / vLLM
| Flag | Default | Description |
|---|---|---|
| `--tensor_parallel_size` | 4 | GPUs per vLLM replica |
| `--num_replicas` | 1 | Single-node replica count (multi-node uses `num_nodes × replicas_per_node`) |
| `--gpus_per_replica` | 4 | GPU slice per replica |
| `--gpu_memory_utilization` | 0.90 | vLLM GPU memory target |
| `--max_num_seqs` | 512 | Max concurrent sequences per engine |
| `--enforce_eager` | off | Disable CUDA graphs. Required for MoE + hybrid-attention models like Qwen3-Coder-Next. Costs 20-40% throughput. |
| `--enable_expert_parallel` | off | Distribute MoE experts across GPUs |
| `--context_window` | 131072 | Max model input length |
| `--cuda_visible_devices` | — | Single-node: GPU list to use (e.g. `"0,1,2,3"`). Omit for multi-node. |
| `--spill_dir` | — | Directory for Ray object spill (single-node only) |

#### Concurrency
| Flag | Default | Description |
|---|---|---|
| `--max_workers` | 256 | Max parallel LLM calls globally |
| `--max_seed_concurrency` | 128 | Max seeds processed in parallel |
| `--max_stage_retries` | 4 | Retry attempts per stage call (exponential backoff) |
| `--limit` | 0 | Process only first N pending seeds (0 = all) |

## Pipeline YAML schema

Top-level:

```yaml
name: <string>           # required, identifier only (not validated to be unique)
flow: <string>           # required, one of: linear, think_execute, or a custom flow name
flow_config: <dict>      # optional, passed to the flow's __init__ (only used by some flows)
stages: [<StageSpec>]    # required for linear; think_execute uses fixed stages
merge: { template: <string> }    # optional, final output assembly for linear flow
```

### StageSpec

```yaml
- name: <string>              # required, identifier
  prompt: <string>            # required (or prompt_file)
  prompt_file: <path>         # alternative to inline prompt (relative to YAML file)
  system_prompt: <string>     # optional
  system_prompt_file: <path>  # alternative
  inputs: [<stage_name>]      # optional, documentation (template vars still work either way)
  fanout: <stage_name>        # optional, run once per item of that prior stage's list output
  decoding:
    temperature: <float>      # default 0.3
    max_tokens: <int>         # default 4096 (stage-dependent)
    parse: json               # optional — parse LLM output as JSON, retry on malformed
```

### Template variables (usable in `prompt` and `system_prompt`)

| Expression | Resolves to |
|---|---|
| `{seed.<field>}` | Any field of the input JSONL record |
| `{seed_id}` | The assigned seed id (explicit or sha1-based) |
| `{<stage_name>}` | Output of a prior stage |
| `{<stage>.<field>}` | Field of a prior JSON-parsed stage |
| `{item}` | Current list element inside a `fanout` stage |

### Merge template (final output)

```yaml
merge:
  template: |
    {seed.text}

    # Generated
    {stage_name_here}
```

When present, after all stages complete, the template is rendered with the full context and written as the output record's `text` field, alongside `seed_id` and `word_count`. If absent, flows may still emit output records directly (e.g. `think_execute` emits one per task).

## Environment variables

| Variable | Purpose | Example |
|---|---|---|
| `HF_HOME` | HuggingFace cache root | `/opt/dlami/nvme` |
| `HF_HUB_CACHE` | HuggingFace hub cache | `/opt/dlami/nvme/hub` |
| `HF_HUB_OFFLINE` | Skip hub lookups (use cache only) | `1` |
| `TRANSFORMERS_OFFLINE` | Skip transformers network calls | `1` |
| `TMPDIR` | Where pip builds | `/tmp` (avoid Lustre for builds) |

When `--ray_address` is set (multi-node), the runner propagates `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` from the head process to all Ray workers automatically.

## Input format

JSONL, one record per line:

```json
{"id": "doc_001", "text": "..."}
{"filename": "something.txt", "text": "..."}
{"text": "no id, will get a sha1 hash"}
```

Only `text` is required. Any other fields are accessible via `{seed.<field>}` in prompts.

## Output format

JSONL, one record per seed (or per task, depending on the flow):

```json
{"seed_id": "doc_001", "text": "...", "word_count": 1234}
```

Flows may add extra fields (e.g. `think_execute` adds `task`).

## Resume behavior

On startup, synthgen reads the output JSONL and collects all `seed_id` values. Any input seed whose id is in that set is skipped. To force re-processing, delete the output file. No CLI flag.

Seeds without an explicit `id` or `filename` get `sha1_<hex>` derived from their `text` — stable across reorders, safe for resume.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success (including nothing-to-do resume) |
| 1 | User error — missing required file, malformed YAML, etc. |
| 2 | `validate` command found errors |

## Files in the repo

| Path | What it is |
|---|---|
| `synthgen/` | The installable package |
| `examples/qa_generation/` | Runnable example: extract + fanout |
| `examples/long_context/` | Runnable example: full think-execute pipeline |
| `scripts/launch_cluster.sh` | Multi-node launcher |
| `tests/smoke/*.sh` | Smoke tests for core functionality |
| `docs/` | This documentation |
