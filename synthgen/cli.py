"""Single-entry CLI: `synthgen run pipeline.yaml --model X --input seeds.jsonl ...`"""
from __future__ import annotations

import argparse
import asyncio
import sys

from . import runner
from .pipeline import Pipeline


def _clean_int(v: str) -> int:
    try:
        return int(str(v).replace(",", "").replace("_", ""))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid integer: {v!r}") from e


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synthgen",
        description="Model-agnostic, pipeline-agnostic synthetic data generation.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a pipeline over a seed file.")
    run.add_argument("pipeline", help="Path to pipeline YAML spec")

    # I/O
    run.add_argument("--input", required=True, help="Input JSONL seed file")
    run.add_argument("--output", required=True, help="Output JSONL file")
    run.add_argument("--intermediate", required=True,
                     help="Debug JSONL (every stage call)")
    run.add_argument("--limit", type=_clean_int, default=0,
                     help="Process first N seeds (0 = all)")

    # Model
    run.add_argument("--model", required=True, help="Model path or HF id")

    # Ray / multi-node
    run.add_argument("--ray_address", default=None,
                     help="'auto' to connect to existing cluster")
    run.add_argument("--num_nodes", type=_clean_int, default=1)
    run.add_argument("--replicas_per_node", type=_clean_int, default=2)
    run.add_argument("--num_replicas", type=_clean_int, default=1,
                     help="Single-node only; multi-node derives from num_nodes * replicas_per_node")
    run.add_argument("--spill_dir", default=None)
    run.add_argument("--cuda_visible_devices", default=None)

    # vLLM
    run.add_argument("--tensor_parallel_size", type=_clean_int, default=4)
    run.add_argument("--gpus_per_replica", type=_clean_int, default=4)
    run.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    run.add_argument("--max_num_seqs", type=_clean_int, default=512)
    run.add_argument("--enforce_eager", action="store_true")
    run.add_argument("--enable_expert_parallel", action="store_true")
    run.add_argument("--context_window", type=_clean_int, default=131072)

    # Concurrency
    run.add_argument("--max_workers", type=_clean_int, default=256)
    run.add_argument("--max_seed_concurrency", type=_clean_int, default=128)
    run.add_argument("--max_stage_retries", type=_clean_int, default=4)

    return p


async def _run(args) -> None:
    pipeline = Pipeline.from_yaml(args.pipeline)
    print(f"[INFO] Pipeline: {pipeline.name} (flow={pipeline.flow_name}, "
          f"{len(pipeline.stages)} stages)")
    await runner.run(args, pipeline)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        asyncio.run(_run(args))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
