# synthgen

Generate synthetic data at scale. You write a YAML describing what the model should do with your seed data; synthgen handles the GPUs, batching, retries, resume, and scaling from 1 GPU to many nodes.

```
seeds.jsonl  +  pipeline.yaml  ──►  synthgen  ──►  output.jsonl
```

## Who this is for

- You have a list of inputs (documents, questions, code snippets, anything with text) and you want an LLM to transform each one into a richer output — summaries, Q&A pairs, long-form answers, critiques, rewrites, whatever.
- You want to run this on **your own GPUs** (1 to N nodes), not pay per-token.
- You don't want to reimplement batching / fault tolerance / resume / multi-node plumbing.

## 5-minute quickstart

```bash
git clone https://github.com/Bharatgen-Tech/syntheticgen.git
cd syntheticgen
pip install -e .

# Try it on a tiny example (3 seeds, produces Q&A pairs):
bash examples/qa_generation/run.sh
```

Output appears at `examples/qa_generation/out/qa.jsonl`. That's it.

Needs: Python 3.10+, a GPU with vLLM-compatible drivers, and the model weights for whatever you're running (`Qwen/Qwen3-Coder-Next` by default).

## Where to go next

| You want to... | Read |
|---|---|
| Install and run your first pipeline end-to-end | [docs/01-getting-started.md](docs/01-getting-started.md) |
| Build a pipeline for your own use case | [docs/02-build-a-pipeline.md](docs/02-build-a-pipeline.md) |
| Run across multiple GPU nodes | [docs/03-scale-up.md](docs/03-scale-up.md) |
| Write custom Python logic beyond YAML | [docs/04-custom-flows.md](docs/04-custom-flows.md) |
| Look up a CLI flag or YAML field | [docs/05-reference.md](docs/05-reference.md) |

## Examples in this repo

- [`examples/qa_generation/`](examples/qa_generation/) — one input → list of concepts → Q&A per concept (demonstrates fanout + JSON parsing)
- [`examples/long_context/`](examples/long_context/) — full long-context pipeline, each seed becomes a 30K-word technical document (demonstrates the pipelined think-execute flow)

## Status

v0.1.0. Core engine + single-node and multi-node validated on H100 clusters. Contributions welcome — see [docs/04-custom-flows.md](docs/04-custom-flows.md).
