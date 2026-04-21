"""Ray-native multi-replica vLLM engine with probe-gated startup."""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional

try:
    import ray
except ImportError:
    ray = None


class InferenceEngine(ABC):
    """Swappable inference backend."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str: ...

    async def start(self) -> None: ...
    async def close(self) -> None: ...


if ray is not None:
    @ray.remote
    class _VLLMReplicaActor:
        """Ray actor wrapping a single AsyncLLMEngine on a fixed set of GPUs."""

        def __init__(self, model_source: str, engine_kwargs: dict,
                     gpu_ids: list[int]):
            import os
            ray_assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if not ray_assigned or ray_assigned == "0,1,2,3,4,5,6,7":
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            actual_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm import AsyncLLMEngine
            engine_args = AsyncEngineArgs(model=model_source, **engine_kwargs)
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
            import inspect, asyncio as _aio
            tokenizer = self._engine.get_tokenizer()
            if inspect.isawaitable(tokenizer) or inspect.iscoroutine(tokenizer):
                try:
                    tokenizer = _aio.get_event_loop().run_until_complete(tokenizer)
                except RuntimeError:
                    tokenizer = _aio.new_event_loop().run_until_complete(tokenizer)
            self._tokenizer = tokenizer
            self._gpu_ids = gpu_ids
            print(f"[REPLICA] Initialized on GPUs {gpu_ids} "
                  f"(CUDA_VISIBLE_DEVICES={actual_gpus})")

        async def probe(self) -> str:
            from vllm import SamplingParams
            import uuid
            sp = SamplingParams(temperature=0.0, max_tokens=1)
            gen = self._engine.generate("ready?", sp, f"warmup-{uuid.uuid4()}")
            async for _ in gen:
                pass
            return f"GPUs={self._gpu_ids}"

        async def generate(
            self,
            system_prompt: str,
            prompt: str,
            max_tokens: int,
            temperature: float = 0.3,
        ):
            from vllm import SamplingParams
            import uuid
            request_id = str(uuid.uuid4())
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                if hasattr(self._tokenizer, "apply_chat_template"):
                    prompt_text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    s_txt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                             if system_prompt else "")
                    prompt_text = (f"{s_txt}<|im_start|>user\n{prompt}"
                                   f"<|im_end|>\n<|im_start|>assistant\n")

                generator = self._engine.generate(prompt_text, sp, request_id)
                result = ""
                num_tokens = 0
                async for output in generator:
                    result = output.outputs[0].text
                    num_tokens = len(output.outputs[0].token_ids)
                return result, num_tokens
            except Exception as e:
                print(f"[ERROR] vLLM generation failed on GPUs {self._gpu_ids}: {e}")
                return "{}", 0
else:
    _VLLMReplicaActor = None


class RayNativeEngine(InferenceEngine):
    """Multi-replica vLLM engine with least-outstanding-requests dispatch,
    NodeAffinity scheduling, probe-gated startup, and replica fault tolerance."""

    def __init__(
        self,
        model_source: str,
        engine_kwargs: dict,
        num_replicas: int = 1,
        gpus_per_replica: int = 4,
        replicas_per_node: int = 2,
        node_ids: list[str] | None = None,
    ):
        if ray is None or _VLLMReplicaActor is None:
            raise RuntimeError("Ray is required for RayNativeEngine")

        self._replicas: list = []
        self._in_flight: list[int] = []
        self._lock: Optional[asyncio.Lock] = None
        self._total_tokens = 0
        self._start_time = time.monotonic()
        self._last_report_time = self._start_time
        self._last_report_tokens = 0
        self._tps_task = None

        total_gpus = num_replicas * gpus_per_replica
        print(f"[ENGINE] Launching {num_replicas} replica(s), "
              f"{gpus_per_replica} GPUs each ({total_gpus} total)")

        if node_ids is not None:
            for node_idx, node_id in enumerate(node_ids):
                for local_replica in range(replicas_per_node):
                    global_idx = node_idx * replicas_per_node + local_replica
                    gpu_ids = list(range(
                        local_replica * gpus_per_replica,
                        (local_replica + 1) * gpus_per_replica,
                    ))
                    actor = _VLLMReplicaActor.options(
                        num_gpus=gpus_per_replica,
                        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                            node_id=node_id,
                            soft=False,
                        ),
                    ).remote(model_source, engine_kwargs, gpu_ids)
                    self._replicas.append(actor)
                    self._in_flight.append(0)
                    print(f"[ENGINE] Replica {global_idx} → node {node_idx} "
                          f"(GPUs {gpu_ids})")
        else:
            for i in range(num_replicas):
                gpu_ids = list(range(i * gpus_per_replica, (i + 1) * gpus_per_replica))
                actor = _VLLMReplicaActor.options(
                    num_gpus=gpus_per_replica,
                ).remote(model_source, engine_kwargs, gpu_ids)
                self._replicas.append(actor)
                self._in_flight.append(0)

    async def _throughput_ticker(self):
        while True:
            await asyncio.sleep(10)
            now = time.monotonic()
            elapsed_last = now - self._last_report_time
            if elapsed_last > 0:
                toks_since_last = self._total_tokens - self._last_report_tokens
                tps_last = toks_since_last / elapsed_last

                total_elapsed = now - self._start_time
                tps_overall = (self._total_tokens / total_elapsed
                               if total_elapsed > 0 else 0)

                in_flight_str = ",".join(str(x) for x in self._in_flight)
                print(f"      [TPS] Speed: {tps_last:,.1f} tok/s | "
                      f"Avg: {tps_overall:,.1f} tok/s | "
                      f"Total: {self._total_tokens:,} tok | "
                      f"InFlight: [{in_flight_str}]")

                self._last_report_time = now
                self._last_report_tokens = self._total_tokens

    async def start(self) -> None:
        self._lock = asyncio.Lock()
        n = len(self._replicas)
        print(f"[ENGINE] Warming up {n} replica(s) — loading weights and "
              f"JIT-compiling kernels (silent until deployed)...")
        t0 = time.monotonic()
        await asyncio.gather(*[r.probe.remote() for r in self._replicas])
        print(f"[ENGINE] All {n} replica(s) deployed in "
              f"{time.monotonic() - t0:,.1f}s — starting pipeline")
        self._start_time = time.monotonic()
        self._last_report_time = self._start_time
        self._last_report_tokens = 0
        self._tps_task = asyncio.create_task(self._throughput_ticker())

    async def close(self) -> None:
        if self._tps_task:
            self._tps_task.cancel()
        for r in self._replicas:
            ray.kill(r)
        self._replicas.clear()

    def _remove_replica(self, idx: int):
        print(f"[WARN] Removing dead replica {idx} "
              f"(remaining: {len(self._replicas) - 1})")
        self._replicas.pop(idx)
        self._in_flight.pop(idx)

    async def generate(
        self,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.3,
    ) -> str:
        for _ in range(3):
            if not self._replicas:
                print("[ERROR] No replicas available.")
                return "{}"

            async with self._lock:
                idx = min(range(len(self._replicas)),
                          key=lambda i: self._in_flight[i])
                self._in_flight[idx] += 1

            try:
                ref = self._replicas[idx].generate.remote(
                    system_prompt, prompt, max_tokens, temperature
                )
                result, num_tokens = await ref
                async with self._lock:
                    self._in_flight[idx] -= 1
                    self._total_tokens += num_tokens
                return result
            except ray.exceptions.RayActorError as e:
                print(f"[ERROR] Replica {idx} died: {e}")
                self._remove_replica(idx)
                continue
            except Exception as e:
                print(f"[ERROR] Ray generate failed: {e}")
                async with self._lock:
                    self._in_flight[idx] -= 1
                return "{}"
        return "{}"
