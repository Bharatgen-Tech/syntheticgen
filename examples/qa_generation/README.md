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

## Files

- `pipeline.yaml` — the pipeline spec
- `seeds.jsonl` — 3 sample documents
- `run.sh` — launches the run on 4 GPUs with sensible defaults

See [docs/02-build-a-pipeline.md](../../docs/02-build-a-pipeline.md) for more pipeline patterns.
