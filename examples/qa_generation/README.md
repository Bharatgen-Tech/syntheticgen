# Q&A generation

**Use this when:** you have source documents and want to generate question/answer pairs grounded in them.

**What it does:** for each input document, extract 3 key concepts, then generate one Q&A pair per concept — 3 seeds in → 3 records out (each with 3 QA pairs inside).

**What it demonstrates:**
- `flow: linear` (most common case)
- `fanout:` — running a stage once per list item in parallel
- `parse: json` — structured output with automatic retry on bad parses

## Run

```bash
bash examples/qa_generation/run.sh
```

Takes ~1-2 minutes after the model loads. Output at `out/qa.jsonl`.

## Adapt to your data

1. Replace [`seeds.jsonl`](seeds.jsonl) with your input documents (one JSON record per line, `text` field required)
2. Edit [`pipeline.yaml`](pipeline.yaml) prompts for your domain
3. Rerun

That's it. No code changes.

## Chunking long documents

Chunking is built into the pipeline — no preprocessing step. Add a `chunking:`
block to [`pipeline.yaml`](pipeline.yaml) and the runner expands each input row
into one or more chunked seeds before stages execute:

```yaml
chunking:
  mode: chunk        # full | chunk | both
  chunk_size: 300    # words per chunk
  overlap: 50        # word overlap between chunks
  field: text        # which field to chunk (default: text)
```

Modes:
- `full` — passthrough, one row → one seed (default; chunking off)
- `chunk` — split each row's `field` into smaller chunks
- `both` — emit the full row AND its chunks (QA at both granularities)

Each chunked seed carries `source_id`, `chunk_index`, `chunk_total` so you can
group results back to the original document. Seed IDs get a `__chunk_NNN`
suffix (and `__full` for the original in `both` mode), so resume still works
correctly across reruns.

## Files

- `pipeline.yaml` — the pipeline spec
- `seeds.jsonl` — 3 sample documents
- `run.sh` — launches the run on 4 GPUs with sensible defaults

See [docs/02-build-a-pipeline.md](../../docs/02-build-a-pipeline.md) for more pipeline patterns.
