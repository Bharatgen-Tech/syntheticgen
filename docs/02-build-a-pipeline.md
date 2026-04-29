# Build a pipeline

A pipeline is a YAML file. It says "for each seed, run these stages, then assemble the results." No Python required for 90% of use cases.

Pick the recipe that matches what you want to do:

| I want to... | Recipe |
|---|---|
| Transform each input into one shorter/longer output | [1. Simple transform](#1-simple-transform) |
| Extract a list from the input, then act on each item | [2. Extract, then fan out](#2-extract-then-fan-out) |
| Generate a draft, critique it, then revise | [3. Draft → critique → revise](#3-draft--critique--revise) |
| Decompose into steps, think about each, then execute | [4. Plan, then step-by-step generate](#4-plan-then-step-by-step-generate) |

Each recipe is a complete, runnable YAML. Copy it, edit prompts, run.

---

## 1. Simple transform

**Use case:** summaries, translations, rewrites, classification — one input → one output.

```yaml
# my_pipelines/summary.yaml
name: summary
flow: linear

stages:
  - name: summary
    system_prompt: "You are a concise technical writer."
    prompt: |
      Summarize the following text in 3-5 sentences:

      {seed.text}
    decoding:
      temperature: 0.3
      max_tokens: 512

merge:
  template: "{summary}"
```

Run it:
```bash
synthgen run my_pipelines/summary.yaml \
  --model Qwen/Qwen3-Coder-Next \
  --input seeds.jsonl \
  --output out/summaries.jsonl \
  --intermediate out/debug.jsonl \
  --num_replicas 1 --tensor_parallel_size 4 --gpus_per_replica 4 \
  --enforce_eager
```

**What to change for your use case:** just the `prompt` and `system_prompt`. If you want structured output (a JSON object instead of prose), add `parse: json` under `decoding:` and change the prompt to ask for JSON.

---

## 2. Extract, then fan out

**Use case:** Q&A generation, multi-topic expansion, "for each keyword, do something" — one input → list of items → one output per item.

```yaml
# my_pipelines/qa.yaml
name: qa_generation
flow: linear

stages:
  - name: concepts
    prompt: |
      List 3 key concepts from this text as a JSON array of strings:

      {seed.text}
    decoding:
      temperature: 0.3
      max_tokens: 256
      parse: json         # parse output as JSON, retry if malformed

  - name: qa_pairs
    fanout: concepts       # run this stage once per concept, in parallel
    prompt: |
      Source text: {seed.text}
      Concept to explain: {item}

      Write a JSON object with "question" and "answer" keys.
    decoding:
      temperature: 0.4
      max_tokens: 512
      parse: json

merge:
  template: |
    Concepts: {concepts}

    QA pairs:
    {qa_pairs}
```

**Key ideas:**
- `parse: json` tells the stage to parse the LLM output as JSON (list or object) — retries on malformed output
- `fanout: concepts` means stage `qa_pairs` runs once per item in the `concepts` list, in parallel
- Inside a fanout stage, `{item}` refers to the current list element
- The result (`qa_pairs`) is itself a list; a later stage could fan out over it

See [`examples/qa_generation/`](../examples/qa_generation/) for the runnable version.

---

## 3. Draft → critique → revise

**Use case:** quality through iteration. Generate something, self-review, fix.

```yaml
# my_pipelines/self_critique.yaml
name: self_critique
flow: linear

stages:
  - name: draft
    prompt: |
      Write a 500-word explanation of:
      {seed.text}
    decoding: { max_tokens: 1024 }

  - name: critique
    prompt: |
      Draft:
      {draft}

      List 3 specific weaknesses of this draft as a JSON array of strings.
    decoding: { max_tokens: 512, parse: json }

  - name: fixes
    fanout: critique       # generate a fix for each weakness in parallel
    prompt: |
      Weakness to fix: {item}
      Original draft: {draft}

      Propose a concrete replacement paragraph.
    decoding: { max_tokens: 512 }

  - name: revised
    prompt: |
      Original draft: {draft}
      Fixes to apply: {fixes}

      Produce a revised version incorporating the fixes.
    decoding: { max_tokens: 1024 }

merge:
  template: "{revised}"
```

**Key ideas:**
- Stages flow top-to-bottom; each stage sees everything produced before it
- You can mix fanout and sequential stages freely

---

## 4. Plan, then step-by-step generate

**Use case:** long-form generation that's too big for one LLM call. Decompose into a plan, then generate each step with accumulated context from prior steps.

This requires the `think_execute` flow instead of `linear`. It's used by [`examples/long_context/`](../examples/long_context/) to generate 30K+ word technical documents per seed.

```yaml
# my_pipelines/long_form.yaml
name: long_form
flow: think_execute     # different executor

flow_config:
  mode: mode2                    # pipelined think-execute (recommended)
  target_output_tokens: 131072   # how big you want the final doc
  context_window: 131072
  instruction_path: ./stage_prompts.yaml   # separate file for stage prompts
  category: YourCategory
```

The `think_execute` flow is more opinionated — it has fixed stages (tasks → plans → per-step think/execute → merge). You configure it via `flow_config` and provide a separate prompts file. See [`examples/long_context/`](../examples/long_context/) for a full working setup.

**When to use this over `linear` + fanout:**
- You need per-step accumulated context (step N sees step N-1's output)
- You want pipelined overlap between think and execute for throughput
- Your final output is very long (>30K words)

Otherwise stick with `linear` — it's simpler and composes better.

---

## Template variables

Inside any `prompt`, `system_prompt`, or `merge.template`:

| Expression | Resolves to |
|---|---|
| `{seed.text}` | The `text` field of the current input record |
| `{seed.id}` or `{seed.<field>}` | Any field of the input record |
| `{seed_id}` | The assigned id (explicit or sha1-hash fallback) |
| `{<stage_name>}` | Output of a prior stage |
| `{<stage>.<field>}` | A field from a prior JSON-parsed stage |
| `{item}` | Current list element inside a `fanout` stage |

## The `merge` section (final output)

After all stages finish, the `linear` flow assembles one output record per seed. `merge.template` controls what goes into the record's `text` field, using the same `{var}` substitution as prompts:

```yaml
merge:
  template: |
    Summary: {summary}
    QA pairs:
    {qa_pairs}
```

Each output JSONL record then looks like:

```json
{"seed_id": "doc_001", "text": "<rendered merge template>", "word_count": 1234}
```

**Key rules:**
- **Omit `merge:` entirely** → the `linear` flow writes **no** output records. Only the intermediate JSONL (per-stage debug log) gets entries. Rarely what you want.
- **`merge.template` has access to everything** — all stage outputs and seed fields are in scope.
- **`think_execute` flow ignores `merge`** — it writes one record per task itself, using its own format. Only `linear` (and custom flows you write) use `merge.template`.
- **Lists stringify naturally** — `{qa_pairs}` will format a list of dicts as Python's `str(...)` does. For structured JSON output, render per-item in a fanout stage instead.

Minimal case — just pass a stage through:

```yaml
merge:
  template: "{summary}"
```

That writes the `summary` stage's output verbatim as the final `text`.

## Validate before you run

```bash
synthgen validate my_pipelines/my_pipeline.yaml
```

Catches: duplicate stage names, unknown flow, fanout pointing at a non-existent stage, template variables referring to unknown names. Cheap; always run it.

## Next step

- Run it locally: [01-getting-started.md](01-getting-started.md)
- Scale it up: [03-scale-up.md](03-scale-up.md)
- Write logic YAML can't express: [04-custom-flows.md](04-custom-flows.md)
