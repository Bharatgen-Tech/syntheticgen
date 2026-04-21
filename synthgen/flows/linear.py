"""Linear flow: stage1 -> stage2 -> ... -> merge.

Supports:
  - Sequential dependencies (later stages see prior stage outputs)
  - Fanout: one stage runs once per item of a prior stage's list output
  - JSON extraction when stage.decoding.parse == 'json'

This is the generic executor. Custom execution orders (think-execute pipelined,
conditional branches) belong in their own flow class.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..pipeline import Pipeline, StageSpec, render_template
from ..utils import extract_json, stage_retry


class LinearFlow:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    async def run_seed(
        self,
        *,
        seed_id: str,
        seed_record: dict,
        engine,
        pipeline: Pipeline,
        output_writer,
        intermediate_writer,
        task_semaphore: asyncio.Semaphore,
        args,
    ) -> None:
        context: dict[str, Any] = {"seed": seed_record, "seed_id": seed_id}

        for stage in pipeline.stages:
            if stage.fanout:
                context[stage.name] = await self._run_fanout(
                    stage, context, engine, task_semaphore,
                    intermediate_writer, seed_id, args,
                )
            else:
                context[stage.name] = await self._run_single(
                    stage, context, engine, task_semaphore,
                    intermediate_writer, seed_id, args,
                )

        if pipeline.merge_template:
            text = render_template(pipeline.merge_template, context)
            record = {
                "seed_id": seed_id,
                "text": text,
                "word_count": len(text.split()),
            }
            await output_writer.write(record)

    async def _run_single(
        self,
        stage: StageSpec,
        context: dict,
        engine,
        task_semaphore: asyncio.Semaphore,
        intermediate_writer,
        seed_id: str,
        args,
    ) -> Any:
        prompt = render_template(stage.prompt, context)
        system_prompt = (
            render_template(stage.system_prompt, context)
            if stage.system_prompt else ""
        )

        async def call():
            async with task_semaphore:
                result = await engine.generate(
                    system_prompt,
                    prompt,
                    max_tokens=stage.decoding.get("max_tokens", 4096),
                    temperature=stage.decoding.get("temperature", 0.3),
                )
            if stage.decoding.get("parse") == "json":
                parsed = extract_json(result)
                return parsed, parsed is not None
            return result, bool(result and result != "{}")

        result = await stage_retry(
            call,
            max_attempts=args.max_stage_retries,
        )

        await intermediate_writer.write({
            "seed_id": seed_id,
            "stage": stage.name,
            "data": result,
        })
        return result

    async def _run_fanout(
        self,
        stage: StageSpec,
        context: dict,
        engine,
        task_semaphore: asyncio.Semaphore,
        intermediate_writer,
        seed_id: str,
        args,
    ) -> list[Any]:
        items = context.get(stage.fanout)
        if not isinstance(items, list):
            raise ValueError(
                f"Stage '{stage.name}' fanout='{stage.fanout}' but that "
                f"prior output is not a list (got {type(items).__name__})."
            )

        async def run_item(item):
            sub_ctx = dict(context)
            sub_ctx[stage.fanout + "_item"] = item
            sub_ctx["item"] = item
            return await self._run_single(
                stage, sub_ctx, engine, task_semaphore,
                intermediate_writer, seed_id, args,
            )

        return await asyncio.gather(*[run_item(it) for it in items])
