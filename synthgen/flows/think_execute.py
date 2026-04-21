"""Think-Execute flow: port of async_lc.py mode1 + mode2.

Pipeline YAML for this flow looks like:

    name: long_context
    flow: think_execute
    flow_config:
      mode: mode2                       # mode1 | mode2
      target_output_tokens: 131072
      context_window: 131072
      instruction_path: instruction.yaml
      category: Code

The flow reads the referenced instruction.yaml (exact same format as current LCG)
to get stage prompts.  All current Stage 1-4 behavior is preserved:
  Stage 1 → tasks list
  Stage 2 → plan per task (all tasks in parallel)
  Stage 3 → think/execute per step (mode1: bulk, mode2: pipelined overlap)
  Stage 4 → merge into [task]/[thinking]/[execution] record
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

import yaml

from ..pipeline import Pipeline
from ..utils import (
    count_words,
    compute_step_budgets,
    truncate_context,
    extract_json,
    repair_truncated_json,
    stage_retry,
)


def _load_prompts(instruction_path: str, category: str) -> dict:
    if not os.path.exists(instruction_path):
        raise FileNotFoundError(f"instruction file not found: {instruction_path}")
    with open(instruction_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if category not in data:
        raise KeyError(
            f"category '{category}' not in {instruction_path}. "
            f"Available: {sorted(data.keys())}"
        )
    required = {"stage1", "stage2", "stage3_think",
                "stage3_execute", "stage3_execute_all"}
    missing = required - set(data[category].keys())
    if missing:
        raise ValueError(f"category '{category}' missing stages: {missing}")
    return data[category]


class ThinkExecuteFlow:
    """Port of the long-context generation pipeline (Stage 1-4)."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        cfg = pipeline.flow_config or {}
        self.mode = cfg.get("mode", "mode2")
        if self.mode not in {"mode1", "mode2"}:
            raise ValueError(f"mode must be mode1 or mode2, got {self.mode!r}")

        self.target_output_tokens = int(cfg.get("target_output_tokens", 131072))
        self.context_window = int(cfg.get("context_window", 131072))

        instr_path = cfg.get("instruction_path")
        if not instr_path:
            raise ValueError("flow_config.instruction_path is required")
        if not os.path.isabs(instr_path):
            instr_path = os.path.join(pipeline._base_dir, instr_path)

        category = cfg.get("category")
        if not category:
            raise ValueError("flow_config.category is required")

        self.prompts = _load_prompts(instr_path, category)
        self.category = category

    # ── Stage helpers ──────────────────────────────────────────────────────
    async def _stage1_tasks(self, engine, seed_text: str,
                            max_retries: int) -> list:
        cfg = self.prompts["stage1"]
        sysp = cfg["system_prompt"].strip()
        prompt = cfg["prompt_template"].format(seed_text=seed_text).strip()

        async def _try():
            resp = await engine.generate(
                sysp, prompt,
                max_tokens=max(16_384, self.context_window // 4))
            parsed = extract_json(repair_truncated_json(resp))
            if isinstance(parsed, list) and parsed:
                return parsed, True
            print(f"[WARN] Stage-1 bad parse. Preview: {resp[:300]}")
            return [], False

        return await stage_retry(_try, max_attempts=max_retries) or []

    async def _stage2_plan(self, engine, seed_text: str, current_task: str,
                           all_tasks: list, max_retries: int) -> list:
        cfg = self.prompts["stage2"]
        sysp = cfg["system_prompt"].strip()
        other_tasks = [t for t in all_tasks if t != current_task]
        prompt = cfg["prompt_template"].format(
            seed_text=seed_text,
            current_task=current_task,
            other_tasks_json=json.dumps(other_tasks, indent=2),
        ).strip()

        async def _try():
            resp = await engine.generate(sysp, prompt, max_tokens=16_384)
            parsed = extract_json(repair_truncated_json(resp))
            if isinstance(parsed, list) and parsed:
                return parsed, True
            print(f"[WARN] Stage-2 bad parse for '{current_task[:60]}'.")
            return [], False

        return await stage_retry(_try, max_attempts=max_retries) or []

    async def _stage3_think(self, engine, seed_text: str, task: str, step: str,
                            accumulated_context: str, previous_step_tail: str,
                            max_tokens: int, max_retries: int) -> str:
        cfg = self.prompts["stage3_think"]
        sysp = cfg["system_prompt"].strip()
        ctx_safe = (truncate_context(accumulated_context, self.context_window)
                    if accumulated_context else "(none — first step)")
        tail_ctx = previous_step_tail or "(none — first step)"
        prompt = cfg["prompt_template"].format(
            seed_text=seed_text,
            task=task,
            step=step,
            accumulated_context=ctx_safe,
            previous_step_tail=tail_ctx,
        ).strip()

        async def _try():
            resp = await engine.generate(sysp, prompt, max_tokens=max_tokens)
            if resp and resp.strip() and resp.strip() != "{}":
                return resp, True
            return "", False

        return await stage_retry(_try, max_attempts=max_retries) or ""

    async def _stage3_execute(self, engine, seed_text: str, task: str, step: str,
                              accumulated_context: str, thinking: str,
                              max_tokens: int, max_retries: int) -> str:
        cfg = self.prompts["stage3_execute"]
        sysp = cfg["system_prompt"].strip()
        ctx_safe = (truncate_context(accumulated_context, self.context_window)
                    if accumulated_context else "(none — first step)")
        prompt = cfg["prompt_template"].format(
            seed_text=seed_text,
            task=task,
            step=step,
            accumulated_context=ctx_safe,
            thinking=thinking,
        ).strip()

        async def _try():
            resp = await engine.generate(sysp, prompt, max_tokens=max_tokens)
            if resp and resp.strip() and resp.strip() != "{}":
                return resp, True
            return "", False

        return await stage_retry(_try, max_attempts=max_retries) or ""

    async def _stage3_execute_bulk(self, engine, seed_text: str, task: str,
                                    plan_steps: list, all_thinking: str,
                                    max_tokens: int, max_retries: int) -> str:
        cfg = self.prompts["stage3_execute_all"]
        sysp = cfg["system_prompt"].strip()
        steps_formatted = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan_steps))
        thinking_safe = truncate_context(all_thinking, self.context_window)
        prompt = cfg["prompt_template"].format(
            seed_text=seed_text,
            task=task,
            steps_formatted=steps_formatted,
            steps_json=steps_formatted,
            all_thinking=thinking_safe,
        ).strip()

        async def _try():
            resp = await engine.generate(sysp, prompt, max_tokens=max_tokens)
            if resp and resp.strip() and resp.strip() != "{}":
                return resp, True
            return "", False

        return await stage_retry(_try, max_attempts=max_retries) or ""

    @staticmethod
    def _merge(seed_id: str, task: str, step_records: list,
               bulk_execution: Optional[str]) -> dict:
        thinking_parts = []
        execution_parts = []
        for sd in step_records:
            idx = sd["step_index"] + 1
            description = sd["step_description"]
            thinking = (sd.get("thinking") or "").strip()
            if thinking:
                thinking_parts.append(
                    f"=== Step {idx}: {description} ===\n{thinking}"
                )
            if bulk_execution is None:
                result = (sd.get("result") or "").strip()
                if result:
                    execution_parts.append(
                        f"=== Step {idx}: {description} ===\n{result}"
                    )

        merged_thinking = "\n\n".join(thinking_parts)
        merged_execution = (bulk_execution.strip() if bulk_execution is not None
                            else "\n\n".join(execution_parts))
        merged = (
            f"[task]:\n{task}\n\n"
            f"[thinking]:\n{merged_thinking}\n\n"
            f"[execution]:\n{merged_execution}"
        )
        return {
            "seed_id": seed_id,
            "task": task,
            "text": merged,
            "word_count": count_words(merged),
        }

    # ── Per-task execution ─────────────────────────────────────────────────
    async def _execute_task_mode1(self, engine, intermediate_writer,
                                   seed_id: str, seed_text: str,
                                   task: str, plan_steps: list,
                                   max_retries: int) -> Optional[dict]:
        print(f" [MODE1 START] Task: {task[:70]}...")
        num_steps = len(plan_steps)
        think_budget, _ = compute_step_budgets(
            target_output_tokens=self.target_output_tokens,
            num_steps=num_steps,
        )
        bulk_execute_budget = self.target_output_tokens

        step_records = []
        accumulated_thinking = ""
        previous_thinking_tail = ""

        for idx, step in enumerate(plan_steps):
            thinking = await self._stage3_think(
                engine, seed_text, task, step,
                accumulated_thinking, previous_thinking_tail,
                max_tokens=think_budget, max_retries=max_retries,
            )
            await intermediate_writer.write({
                "seed_id": seed_id,
                "stage": f"stage3_thinking_step_{idx}",
                "data": {"step_index": idx, "step_description": step,
                         "thinking": thinking},
            })
            step_records.append({
                "step_index": idx, "step_description": step,
                "thinking": thinking, "result": "",
            })
            accumulated_thinking += (
                f"\n--- Thinking for Step {idx+1}: {step} ---\n{thinking}\n"
            )
            lines = thinking.splitlines()
            previous_thinking_tail = "\n".join(lines[-20:]) if lines else ""

        bulk_result = await self._stage3_execute_bulk(
            engine, seed_text, task, plan_steps,
            accumulated_thinking,
            max_tokens=bulk_execute_budget, max_retries=max_retries,
        )
        await intermediate_writer.write({
            "seed_id": seed_id,
            "stage": "stage3_execute_all",
            "data": {"task": task, "steps": plan_steps, "result": bulk_result},
        })
        print(f" [MODE1 DONE]  Task: {task[:70]}...")
        return self._merge(seed_id, task, step_records, bulk_execution=bulk_result)

    async def _execute_task_mode2(self, engine, intermediate_writer,
                                   seed_id: str, seed_text: str,
                                   task: str, plan_steps: list,
                                   max_retries: int) -> Optional[dict]:
        print(f" [MODE2 START] Task: {task[:70]}...")
        num_steps = len(plan_steps)
        think_budget, execute_budget = compute_step_budgets(
            target_output_tokens=self.target_output_tokens,
            num_steps=num_steps,
        )

        accumulated_context = ""
        previous_step_tail = ""
        step_records = []
        pending_thinking = None

        for idx, step in enumerate(plan_steps):
            is_last = (idx == num_steps - 1)

            if pending_thinking is not None:
                thinking = await pending_thinking
                pending_thinking = None
            else:
                thinking = await self._stage3_think(
                    engine, seed_text, task, step,
                    accumulated_context, previous_step_tail,
                    max_tokens=think_budget, max_retries=max_retries,
                )

            await intermediate_writer.write({
                "seed_id": seed_id,
                "stage": f"stage3_thinking_step_{idx}",
                "data": {"step_index": idx, "step_description": step,
                         "thinking": thinking},
            })

            exec_coro = self._stage3_execute(
                engine, seed_text, task, step,
                accumulated_context, thinking,
                max_tokens=execute_budget, max_retries=max_retries,
            )

            if not is_last:
                next_step = plan_steps[idx + 1]
                thinking_lines = thinking.splitlines()
                thinking_tail = ("\n".join(thinking_lines[-20:])
                                 if thinking_lines else "")
                think_next_coro = self._stage3_think(
                    engine, seed_text, task, next_step,
                    accumulated_context, thinking_tail,
                    max_tokens=think_budget, max_retries=max_retries,
                )
                exec_task = asyncio.create_task(exec_coro)
                think_task = asyncio.create_task(think_next_coro)
                result = await exec_task
                pending_thinking = think_task
            else:
                result = await exec_coro

            await intermediate_writer.write({
                "seed_id": seed_id,
                "stage": f"stage3_execute_step_{idx}",
                "data": {"step_index": idx, "step_description": step,
                         "result": result},
            })
            step_records.append({
                "step_index": idx, "step_description": step,
                "thinking": thinking, "result": result,
            })
            accumulated_context += (
                f"\n--- Output of Step {idx+1}: {step} ---\n{result}\n"
            )
            lines = result.splitlines()
            previous_step_tail = "\n".join(lines[-20:]) if lines else ""

        print(f" [MODE2 DONE]  Task: {task[:70]}...")
        return self._merge(seed_id, task, step_records, bulk_execution=None)

    async def _execute_task(self, engine, intermediate_writer,
                             seed_id: str, seed_text: str,
                             task: str, plan_steps: list,
                             max_retries: int) -> Optional[dict]:
        if self.mode == "mode1":
            return await self._execute_task_mode1(
                engine, intermediate_writer, seed_id, seed_text,
                task, plan_steps, max_retries)
        return await self._execute_task_mode2(
            engine, intermediate_writer, seed_id, seed_text,
            task, plan_steps, max_retries)

    # ── Entry point ────────────────────────────────────────────────────────
    async def run_seed(
        self,
        *,
        seed_id: str,
        seed_record: dict,
        engine,
        pipeline,
        output_writer,
        intermediate_writer,
        task_semaphore: asyncio.Semaphore,
        args,
    ) -> None:
        seed_text = seed_record.get("text", "")
        max_retries = args.max_stage_retries
        print(f"\n{'='*60}\n[SEED] {seed_id}  [mode={self.mode}]")

        tasks = await self._stage1_tasks(engine, seed_text, max_retries)
        if not tasks:
            print(f"  [SKIP] Stage-1 returned no tasks for '{seed_id}'.")
            return
        await intermediate_writer.write({
            "seed_id": seed_id, "stage": "stage1_tasks",
            "data": {"tasks": tasks},
        })

        async def plan_one_task(task: str):
            plan = await self._stage2_plan(
                engine, seed_text, task, tasks, max_retries)
            if not plan:
                return None
            await intermediate_writer.write({
                "seed_id": seed_id, "stage": "stage2_plan",
                "data": {"task": task, "plan": plan},
            })
            return {"task": task, "plan": plan}

        plan_results = await asyncio.gather(*[plan_one_task(t) for t in tasks])
        task_plan_pairs = [r for r in plan_results if r is not None]
        if not task_plan_pairs:
            print(f"  [SKIP] No valid task-plan pairs for '{seed_id}'.")
            return

        async def bounded_execute(tp: dict):
            async with task_semaphore:
                return await self._execute_task(
                    engine, intermediate_writer,
                    seed_id, seed_text, tp["task"], tp["plan"],
                    max_retries)

        results = await asyncio.gather(
            *[bounded_execute(tp) for tp in task_plan_pairs],
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                print(f" [ERROR] Task raised: {res}")
                continue
            if res and res.get("text", "").strip():
                await output_writer.write(res)
                print(f" [SAVED] seed={seed_id} | words={res.get('word_count', 0):,}")
