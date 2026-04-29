# Long-context generation

**Use this when:** you need very long-form generation per seed (30K-130K words) — technical walkthroughs, deep explanations, multi-part guides. Too big for a single LLM call.

**What it does:** for each seed, generate 10 self-contained tasks, plan each one into steps, then for every step run pipelined think-execute with accumulated context from prior steps. Each task produces a 30-50K word document.

**What it demonstrates:**
- `flow: think_execute` — the pipelined long-form flow
- External prompts via `flow_config.instruction_path`
- Per-step token budgeting (`compute_step_budgets` under the hood)

## Run

```bash
bash examples/long_context/run.sh            # 1 seed, ~2 hours
bash examples/long_context/run.sh 5          # 5 seeds, longer
```

Output at `out/corpus.jsonl`. Takes hours because each seed produces ~10 tasks × ~130K tokens.

For a fast smoke test of the infra (not real throughput), use `--limit 1` on 8 GPUs — you'll see the flow work end-to-end in ~5-10 minutes.

## Adapt to your data

1. Replace `seeds.jsonl` with your inputs
2. Edit `instruction.yaml` to change the stage prompts (5 stages: `stage1`, `stage2`, `stage3_think`, `stage3_execute`, `stage3_execute_all`) or add a new category alongside `Code`
3. Change `flow_config.category` in `pipeline.yaml` to match
4. Rerun

## Files

- `pipeline.yaml` — flow config (mode, target tokens, instruction path)
- `instruction.yaml` — 5-stage prompt templates per category (Code, Stem, Maths, Mythology, Law)
- `seeds.jsonl` — 3 sample codechef problems
- `run.sh` — launches on 8 GPUs (2 replicas, TP=4)

## Scale out

Single-node maxes around 1,600 tok/s. For faster generation, use the multi-node launcher with this pipeline — see [docs/03-scale-up.md](../../docs/03-scale-up.md).
