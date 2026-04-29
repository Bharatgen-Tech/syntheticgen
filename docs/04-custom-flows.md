# Custom flows

YAML handles most cases. But when you need control flow — conditional branching, retrieval hooks, multi-turn loops, pipelined overlap — you write a Python **flow** class.

## When do I need this?

| Your pipeline needs... | Use |
|---|---|
| Stages flow top-to-bottom | `linear` (YAML) |
| For each item of a prior stage, do something | `linear` with `fanout:` (YAML) |
| If stage-N output contains X, go to stage A, else stage B | Custom flow |
| Pipelined overlap between stages (e.g. think + execute) | Custom flow (`think_execute` is a built-in example) |
| RAG retrieval before a stage's prompt | Custom flow |
| A loop that runs until some condition | Custom flow |

If your case is in the top two rows, skip this page — see [02-build-a-pipeline.md](02-build-a-pipeline.md).

## The contract

A flow is a class with one method:

```python
async def run_seed(
    self,
    *,
    seed_id: str,
    seed_record: dict,
    engine,                # has engine.generate(system, prompt, max_tokens, temperature)
    pipeline,              # the parsed YAML (stages, flow_config, merge_template)
    output_writer,         # .write(dict)
    intermediate_writer,   # .write(dict) — debug log
    task_semaphore,        # bound concurrent LLM calls; wrap engine.generate in it
    args,                  # CLI args (max_stage_retries, etc)
) -> None:
    ...
```

The runner calls this once per input seed. You decide what happens in between, and write output records via `output_writer.write(...)`.

## Minimum example

```python
# synthgen/flows/my_flow.py
from __future__ import annotations
import asyncio
from ..pipeline import Pipeline, render_template
from ..utils import stage_retry


class MyFlow:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        # Read flow-specific config if you need it:
        # self.target_tokens = pipeline.flow_config.get("target_tokens", 4096)

    async def run_seed(
        self, *, seed_id, seed_record, engine, pipeline,
        output_writer, intermediate_writer, task_semaphore, args,
    ):
        context = {"seed": seed_record, "seed_id": seed_id}

        # Run stages however you want. Here's a simple sequential loop
        # (the same as LinearFlow — replace with your own logic):
        for stage in pipeline.stages:
            prompt = render_template(stage.prompt, context)
            sysp = render_template(stage.system_prompt or "", context)

            async def call():
                async with task_semaphore:
                    resp = await engine.generate(
                        sysp, prompt,
                        max_tokens=stage.decoding.get("max_tokens", 2048),
                        temperature=stage.decoding.get("temperature", 0.3),
                    )
                return resp, bool(resp and resp != "{}")

            result = await stage_retry(call, max_attempts=args.max_stage_retries)
            context[stage.name] = result

            await intermediate_writer.write({
                "seed_id": seed_id, "stage": stage.name, "data": result,
            })

        # Emit the final record:
        if pipeline.merge_template:
            text = render_template(pipeline.merge_template, context)
            await output_writer.write({
                "seed_id": seed_id,
                "text": text,
                "word_count": len(text.split()),
            })
```

## Register it

Edit [`synthgen/flows/__init__.py`](../synthgen/flows/__init__.py):

```python
from .linear import LinearFlow
from .think_execute import ThinkExecuteFlow
from .my_flow import MyFlow                 # add

FLOWS = {
    "linear": LinearFlow,
    "think_execute": ThinkExecuteFlow,
    "my_flow": MyFlow,                       # add
}
```

## Use it from a pipeline YAML

```yaml
name: pipeline_that_uses_my_flow
flow: my_flow
flow_config:
  target_tokens: 8192         # read in MyFlow.__init__ via pipeline.flow_config
stages:
  - name: foo
    prompt: "..."
    decoding: { max_tokens: 2048 }
```

## Things to use (don't reinvent)

| Helper | From | Does |
|---|---|---|
| `render_template(str, context)` | `synthgen.pipeline` | `{var}` and `{var.field}` substitution in prompts |
| `stage_retry(async_fn, max_attempts)` | `synthgen.utils` | Retry a callable with exponential backoff |
| `extract_json(text)` | `synthgen.utils` | Safely parse LLM output as JSON (handles code fences, truncation) |
| `truncate_context(text, context_window)` | `synthgen.utils` | Keep tail of text under token budget (word-based estimate) |
| `compute_step_budgets(target, num_steps)` | `synthgen.utils` | Divide a token budget across N steps (used by `think_execute`) |

## What NOT to do in a flow

| Don't | Use instead |
|---|---|
| Call `ray.init()` / manage the cluster | Runner handles it — use the `engine` handle |
| Write directly to files | Use the `output_writer` / `intermediate_writer` (they're buffered, resume-safe) |
| Create your own semaphore / connection pool | Use `task_semaphore` passed in |
| Handle seed loading, resume, retries across seeds | Runner handles those above `run_seed` |

If you find yourself doing any of the above, step back — you're rebuilding infrastructure the runner already provides.

## Reference flow: `think_execute`

The built-in [`flows/think_execute.py`](../synthgen/flows/think_execute.py) is a good reference for a complex custom flow. It demonstrates:
- Reading prompts from an external file via `flow_config.instruction_path`
- Nested concurrency (asyncio tasks for pipelined overlap within a seed)
- Accumulated context with truncation across steps
- Token budgeting (`compute_step_budgets`)

## Next step

- Look up CLI flags or YAML fields: [05-reference.md](05-reference.md)
