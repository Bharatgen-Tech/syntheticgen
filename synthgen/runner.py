"""Top-level orchestration: Ray init, multi-node wait, seed loading, resume,
flow dispatch, output writer lifecycle.

The runner is flow-agnostic — it just calls flow.run(seed, ctx) for each seed.
Flows are plugged in from synthgen.flows.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Optional

from .engine import InferenceEngine, RayNativeEngine
from .writer import AsyncBufferedWriter
from .pipeline import Pipeline

try:
    import ray
except ImportError:
    ray = None


def _init_ray(args) -> None:
    """Initialize Ray: local instance, or attach to existing cluster."""
    if ray is None:
        raise RuntimeError("Ray is required. Install with: pip install ray")

    runtime_env: Optional[dict] = None
    if args.ray_address:
        env_vars = {
            "PATH": os.path.expanduser("~/miniconda3/bin") + ":"
                    + os.environ.get("PATH", "/usr/bin:/bin"),
        }
        # Propagate HF cache + offline flags so workers use local NVMe
        for key in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE", "TRANSFORMERS_CACHE"):
            if os.environ.get(key):
                env_vars[key] = os.environ[key]
        runtime_env = {"env_vars": env_vars}
    elif args.cuda_visible_devices:
        runtime_env = {
            "env_vars": {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}
        }

    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    if args.ray_address:
        ray_kwargs["address"] = args.ray_address
    if runtime_env is not None:
        ray_kwargs["runtime_env"] = runtime_env
    if args.spill_dir and not args.ray_address:
        ray_kwargs["_system_config"] = {
            "object_spilling_config": json.dumps({
                "type": "filesystem",
                "params": {"directory_path": args.spill_dir},
            })
        }

    ray.init(**ray_kwargs)
    print("[INFO] Ray initialized")
    print(f"[INFO] Available resources: {ray.available_resources()}")


def _wait_for_nodes(num_nodes: int) -> list[str]:
    """Wait for N nodes to join. Returns list of alive NodeIDs."""
    print(f"[INFO] Waiting for {num_nodes} node(s) to join the cluster ...")
    for attempt in range(120):
        alive = [n for n in ray.nodes() if n.get("Alive", False)]
        if len(alive) >= num_nodes:
            break
        if attempt % 6 == 0:
            print(f"[INFO]   ... {len(alive)}/{num_nodes} nodes alive")
        time.sleep(5)
    else:
        alive = [n for n in ray.nodes() if n.get("Alive", False)]
        raise RuntimeError(
            f"Only {len(alive)}/{num_nodes} nodes joined after 10 minutes."
        )
    print(f"[INFO] All {num_nodes} node(s) alive:")
    for n in alive[:num_nodes]:
        print(f"       - {n.get('NodeManagerAddress', '?')}: "
              f"GPUs={n.get('Resources', {}).get('GPU', 0)}")
    return [n["NodeID"] for n in alive[:num_nodes]]


def _build_engine(args) -> RayNativeEngine:
    """Construct RayNativeEngine with all vLLM flags + multi-node pinning."""
    engine_kwargs: dict[str, Any] = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.context_window,
        "max_num_seqs": args.max_num_seqs,
        "enable_expert_parallel": args.enable_expert_parallel,
    }

    node_ids: Optional[list[str]] = None
    if args.num_nodes > 1:
        node_ids = _wait_for_nodes(args.num_nodes)
        total_replicas = args.num_nodes * args.replicas_per_node
        print(f"[INFO] Multi-node: {args.num_nodes} nodes × "
              f"{args.replicas_per_node} replicas/node = "
              f"{total_replicas} total replicas")
    else:
        total_replicas = args.num_replicas

    return RayNativeEngine(
        model_source=args.model,
        engine_kwargs=engine_kwargs,
        num_replicas=total_replicas,
        gpus_per_replica=args.gpus_per_replica,
        replicas_per_node=args.replicas_per_node,
        node_ids=node_ids,
    )


def _load_seeds(input_path: str) -> list[tuple[str, dict]]:
    """Load seeds from JSONL. Returns [(seed_id, record), ...]."""
    seeds: list[tuple[str, dict]] = []
    seen_ids: dict[str, int] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Malformed JSON on line {line_num}: {e}")
                continue
            text = record.get("text", "")
            if not text.strip():
                print(f"[WARN] Line {line_num} has empty text — skipping.")
                continue
            raw_id = (record.get("filename")
                      or record.get("id")
                      or "sha1_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])
            n = seen_ids.get(raw_id, 0)
            seen_ids[raw_id] = n + 1
            seed_id = raw_id if n == 0 else f"{raw_id}__{n}"
            seeds.append((seed_id, record))
    return seeds


def _load_completed(output_path: str) -> set[str]:
    """Read output JSONL and return seed_ids already processed."""
    completed: set[str] = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("seed_id"):
                    completed.add(r["seed_id"])
            except json.JSONDecodeError:
                pass
    return completed


async def run(args, pipeline: Pipeline) -> None:
    """End-to-end pipeline execution.

    Flow-agnostic: `pipeline` provides the stage spec + a flow instance that
    knows how to execute a single seed.
    """
    # ── Ray + engine ───────────────────────────────────────────────────────
    _init_ray(args)
    engine = _build_engine(args)
    await engine.start()

    # ── I/O setup ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.intermediate)), exist_ok=True)
    if not os.path.exists(args.intermediate):
        open(args.intermediate, "w").close()

    output_writer = AsyncBufferedWriter(args.output)
    intermediate_writer = AsyncBufferedWriter(args.intermediate)
    await output_writer.start()
    await intermediate_writer.start()

    # ── Seeds + resume ─────────────────────────────────────────────────────
    seeds = _load_seeds(args.input)
    print(f"[INFO] Loaded {len(seeds):,} seeds from '{args.input}'")

    completed = _load_completed(args.output)
    if completed:
        print(f"[INFO] Resuming — {len(completed):,} seed(s) already complete.")

    pending = [(sid, rec) for sid, rec in seeds if sid not in completed]
    if args.limit and args.limit > 0:
        pending = pending[:args.limit]
        print(f"[INFO] Limited run — first {args.limit} pending seed(s).")

    print(f"[INFO] {len(pending):,} seed(s) to process.")
    if not pending:
        await output_writer.close()
        await intermediate_writer.close()
        await engine.close()
        return

    # ── Run pipeline per seed ──────────────────────────────────────────────
    seed_semaphore = asyncio.Semaphore(args.max_seed_concurrency)
    task_semaphore = asyncio.Semaphore(args.max_workers)

    async def bounded_seed(seed_id: str, record: dict):
        async with seed_semaphore:
            try:
                await pipeline.flow.run_seed(
                    seed_id=seed_id,
                    seed_record=record,
                    engine=engine,
                    pipeline=pipeline,
                    output_writer=output_writer,
                    intermediate_writer=intermediate_writer,
                    task_semaphore=task_semaphore,
                    args=args,
                )
            except Exception as exc:
                print(f"[ERROR] Seed '{seed_id}' failed: {exc}")

    try:
        await asyncio.gather(*[bounded_seed(s, r) for s, r in pending])
    finally:
        await output_writer.close()
        await intermediate_writer.close()
        await engine.close()

    print("\n[OK] Pipeline execution complete.")
