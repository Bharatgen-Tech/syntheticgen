# Scale up

From 1 GPU to N nodes with the same pipeline. No code changes — same YAML, more replicas.

## Single-node baseline (you already did this)

```bash
synthgen run my_pipeline.yaml \
  --model Qwen/Qwen3-Coder-Next \
  --input seeds.jsonl \
  --output out/corpus.jsonl \
  --intermediate out/debug.jsonl \
  --num_replicas 2 \              # two vLLM engines on this node
  --tensor_parallel_size 4 \      # each engine uses 4 GPUs (gpus_per_replica defaults to this)
  --gpu_memory_utilization 0.92 \
  --max_workers 256 \
  --max_seed_concurrency 128 \
  --enforce_eager                  # required for MoE + hybrid attention models
```

On 8× H100, expect ~1,600 tok/s.

## Multi-node (one command)

Edit the IPs at the top of [`scripts/launch_cluster.sh`](../scripts/launch_cluster.sh):

```bash
HEAD_IP="10.0.1.10"
WORKER_IPS=(
  "10.0.1.11"
  "10.0.1.12"
)
```

Then from the head node, **one command:**

```bash
bash scripts/launch_cluster.sh head 100
```

That `100` is the seed limit. The launcher does everything:

1. Purges orphaned Ray/vLLM processes on every node
2. Runs `nvidia-smi` on each node and **aborts cleanly if any GPU belongs to another user's workload** (won't stomp on someone else's run)
3. `rsync`s the model to each worker's local NVMe (skips files already there)
4. Starts Ray head → SSHes to workers → joins them
5. Runs the pipeline with 4 replicas (2 per node × 2 nodes)
6. Gracefully shuts down and verifies all GPUs are freed

**Prerequisites:**
- Passwordless SSH from head to every worker (you can `ssh worker-ip` without typing a password)
- Shared `/home` (or whatever your miniconda is on) — one `pip install -e .` should be visible on every node
- Same Python + Ray + vLLM versions on all nodes

## Expected throughput

| Setup | Throughput | Notes |
|---|---|---|
| 1 node, 8 GPUs, 2 replicas | ~1,600 tok/s | Steady-state with ~100+ concurrent seeds |
| 2 nodes, 16 GPUs, 4 replicas | ~5,000 tok/s | Linear-ish scaling |
| 4 nodes, 32 GPUs, 8 replicas | ~10,000 tok/s | Projected |
| 8 nodes, 64 GPUs, 16 replicas | ~20,000 tok/s | Projected |

**The catch — feed the replicas enough work.** Each replica wants a dense batch. If you have 4 replicas and process 4 seeds, each replica only has 1 seed, which is nowhere near its capacity and throughput collapses. Rule of thumb:

> **max_seed_concurrency ≥ 4 × (total replicas)**

So for 4 replicas, `--max_seed_concurrency 256` is fine; for 16 replicas, bump to `--max_seed_concurrency 512+`. If you don't have enough input seeds to fill that, the multi-node run won't be faster than single-node — same amount of work spread across more resources.

## Monitoring

While the launcher runs, watch `[TPS]` lines in its output:

```
[TPS] Speed: 5,200 tok/s | Avg: 4,800 tok/s | Total: 28M tok | InFlight: [220,218,221,219]
                                                                             ^^^^ balanced = healthy
```

- `Speed` is the tok/s in the last 10 seconds
- `InFlight` is one number per replica — should be balanced. Unbalanced (e.g. `[200, 200, 0, 0]`) means 2 of 4 replicas are idle — usually a startup artifact, should equalize within a minute

To check GPU usage on a node:
```bash
ssh worker-ip "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

## Resume across preemptions

Resume is on by default. If the cluster dies mid-run, re-running the exact same command skips seeds already in the output and picks up where you left off. See [01-getting-started.md](01-getting-started.md) step 5.

## Troubleshooting multi-node

| Symptom | Cause | Fix |
|---|---|---|
| Preflight aborts with "GPUs busy" | Other user has processes on the node | Pick different nodes. **Don't lower the 5000 MiB threshold** — you'd be stomping on their work |
| Worker loads weights slowly (80s/shard) | Worker doesn't have model on local NVMe, reading over NFS | Happens once on first run per worker. The launcher's `rsync` step stages the model; subsequent runs skip |
| Worker can't find model at all | First run, rsync hasn't run yet | Let the launcher do its full sequence — it's idempotent |
| "Only 1/2 nodes joined after 10 minutes" | Worker can't reach head's port 6379 | Check: `ssh worker curl HEAD_IP:6379`. Firewall or wrong IP |
| `Free memory on device cuda:0 (37/81 GiB)` | Leftover vLLM workers holding GPU memory | The launcher's `pkill` at startup handles this — if testing manually: `pkill -9 -f 'vllm\|EngineCore\|ray::'` on the affected node |
| Ray tries to connect to a dead cluster address | Stale `/tmp/ray/ray_current_cluster` from a prior run | Runner now forces `address="local"` when no `--ray_address` — but if you ran `ray start` manually and killed it roughly: `rm -rf /tmp/ray/ray_current_cluster /tmp/ray/session_*` |
| Throughput low despite GPUs being utilized | Underfed — `max_seed_concurrency` too low for the number of replicas | Raise `--max_seed_concurrency` (rule: `≥ 4 × replicas`); or run more seeds |

## What the launcher does (if you want to read the code)

Everything in [`scripts/launch_cluster.sh`](../scripts/launch_cluster.sh) is idempotent and observable — it prints what it's doing at every step. If you hit an error, reading the last ~50 lines of its output usually tells you exactly where.

## Next step

- Customize the execution logic: [04-custom-flows.md](04-custom-flows.md)
- Look up a flag: [05-reference.md](05-reference.md)
