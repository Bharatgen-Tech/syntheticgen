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

    val = sub.add_parser(
        "validate",
        help="Check a pipeline YAML for errors without running it.",
    )
    val.add_argument("pipeline", help="Path to pipeline YAML spec")

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


def _validate(path: str) -> int:
    """Load a pipeline YAML and check for common errors. Returns 0 on success."""
    import re
    from .flows import FLOWS

    errors: list[str] = []
    warnings: list[str] = []

    try:
        pipeline = Pipeline.from_yaml(path)
    except Exception as e:
        print(f"[FAIL] Could not load '{path}': {e}")
        return 2

    print(f"Loaded '{pipeline.name}' (flow={pipeline.flow_name}, "
          f"{len(pipeline.stages)} stage(s))")

    # 1. Flow must be registered.
    if pipeline.flow_name not in FLOWS:
        errors.append(
            f"flow '{pipeline.flow_name}' is not registered. "
            f"Available: {sorted(FLOWS.keys())}"
        )

    # 2. Instantiate the flow (catches flow_config errors like missing instruction_path).
    try:
        _ = pipeline.flow
    except Exception as e:
        errors.append(f"flow '{pipeline.flow_name}' failed to initialize: {e}")

    # 3. Stage-level checks.
    stage_names = [s.name for s in pipeline.stages]
    dup = {n for n in stage_names if stage_names.count(n) > 1}
    if dup:
        errors.append(f"duplicate stage names: {sorted(dup)}")

    seen: set[str] = set()
    for stage in pipeline.stages:
        if not stage.name:
            errors.append("stage with no name")
            continue

        # fanout must reference an earlier stage.
        if stage.fanout and stage.fanout not in seen:
            errors.append(
                f"stage '{stage.name}' fanout='{stage.fanout}' but no prior stage "
                f"by that name (prior: {sorted(seen) or '[none]'})"
            )

        # Template vars that look like {stage_name} must reference seeds, items, or prior stages.
        known = set(seen) | {"seed", "seed_id", "item"}
        refs = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)", stage.prompt or ""))
        if stage.system_prompt:
            refs |= set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)", stage.system_prompt))
        for ref in refs:
            if ref not in known and ref != stage.name:
                warnings.append(
                    f"stage '{stage.name}' references '{{{ref}}}' but no prior stage, "
                    f"seed, or item provides it. Known: {sorted(known)}"
                )

        # Decoding sanity.
        mt = stage.decoding.get("max_tokens", 0)
        if mt and (mt < 1 or mt > 1_000_000):
            warnings.append(f"stage '{stage.name}' max_tokens={mt} looks wrong")
        parse = stage.decoding.get("parse")
        if parse and parse != "json":
            warnings.append(
                f"stage '{stage.name}' decoding.parse='{parse}' is not recognized "
                f"(supported: 'json')"
            )

        seen.add(stage.name)

    # 4. merge template references.
    if pipeline.merge_template:
        merge_refs = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)",
                                    pipeline.merge_template))
        known = set(seen) | {"seed", "seed_id"}
        for ref in merge_refs:
            if ref not in known:
                warnings.append(
                    f"merge template references '{{{ref}}}' but no stage produces it. "
                    f"Known: {sorted(known)}"
                )

    # ── report ───────────────────────────────────────────────────────────────
    for w in warnings:
        print(f"  [WARN] {w}")
    for e in errors:
        print(f"  [FAIL] {e}")

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"\nOK ({len(warnings)} warning(s))")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        asyncio.run(_run(args))
        return 0
    if args.cmd == "validate":
        return _validate(args.pipeline)
    return 1


if __name__ == "__main__":
    sys.exit(main())
