"""Pipeline spec: YAML-declared stages + pluggable flow.

A Pipeline is a YAML file that declares:
  - name: identifier
  - flow: which executor to use (linear | think_execute | ...)
  - stages: list of stage specs (name, prompt, inputs, decoding, fanout)
  - merge: optional final-output template

Flows consume the spec and decide how to execute (sequential, pipelined,
conditional, etc).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class StageSpec:
    """Declarative spec for a single stage."""
    name: str
    prompt: str                       # template string; may contain {var}
    inputs: list[str] = field(default_factory=list)  # prior outputs to pass in
    fanout: Optional[str] = None      # if set, run once per item in that prior output
    decoding: dict = field(default_factory=dict)     # temperature, max_tokens
    system_prompt: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict, base_dir: str) -> "StageSpec":
        prompt = d.get("prompt")
        if not prompt and d.get("prompt_file"):
            path = d["prompt_file"]
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            with open(path, "r", encoding="utf-8") as f:
                prompt = f.read()
        if prompt is None:
            raise ValueError(f"Stage '{d.get('name')}' needs prompt or prompt_file")

        system_prompt = d.get("system_prompt")
        if not system_prompt and d.get("system_prompt_file"):
            path = d["system_prompt_file"]
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            with open(path, "r", encoding="utf-8") as f:
                system_prompt = f.read()

        return cls(
            name=d["name"],
            prompt=prompt,
            inputs=d.get("inputs", []),
            fanout=d.get("fanout"),
            decoding=d.get("decoding", {}),
            system_prompt=system_prompt,
        )


@dataclass
class ChunkingSpec:
    """Optional input-text chunking applied by the runner before stages execute.

    mode:
      full   no chunking (default; equivalent to omitting the block)
      chunk  split each seed's `field` into smaller pieces
      both   emit the original seed AND its chunks

    Sizes are in whitespace-separated words. Overlap must be < chunk_size.
    """
    mode: str = "full"
    chunk_size: int = 300
    overlap: int = 50
    field: str = "text"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ChunkingSpec":
        if not d:
            return cls()
        mode = d.get("mode", "full")
        if mode not in ("full", "chunk", "both"):
            raise ValueError(f"chunking.mode must be full|chunk|both, got '{mode}'")
        size = int(d.get("chunk_size", 300))
        overlap = int(d.get("overlap", 50))
        if mode != "full":
            if size <= 0:
                raise ValueError("chunking.chunk_size must be > 0")
            if overlap < 0 or overlap >= size:
                raise ValueError("chunking.overlap must be >= 0 and < chunk_size")
        return cls(mode=mode, chunk_size=size, overlap=overlap, field=d.get("field", "text"))


@dataclass
class Pipeline:
    name: str
    flow_name: str
    stages: list[StageSpec]
    merge_template: Optional[str]
    flow_config: dict
    chunking: ChunkingSpec
    _base_dir: str

    _flow_cached: Any = None

    @property
    def flow(self):
        """Lazy-instantiate the flow to avoid circular imports at module load."""
        if self._flow_cached is None:
            from .flows import FLOWS
            if self.flow_name not in FLOWS:
                raise ValueError(
                    f"Unknown flow '{self.flow_name}'. Available: {sorted(FLOWS.keys())}"
                )
            object.__setattr__(self, "_flow_cached", FLOWS[self.flow_name](self))
        return self._flow_cached

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        base_dir = os.path.dirname(os.path.abspath(path))
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        stages = [StageSpec.from_dict(s, base_dir) for s in data.get("stages", [])]
        merge = data.get("merge", {})
        merge_template = merge.get("template") if isinstance(merge, dict) else None

        return cls(
            name=data["name"],
            flow_name=data.get("flow", "linear"),
            stages=stages,
            merge_template=merge_template,
            flow_config=data.get("flow_config", {}),
            chunking=ChunkingSpec.from_dict(data.get("chunking")),
            _base_dir=base_dir,
        )


def render_template(template: str, context: dict) -> str:
    """Simple {var} and {nested.field} substitution. No expressions, no loops.

    Dicts and lists are JSON-serialized (rather than str(repr)) so structured
    stage outputs can be passed through merge templates as valid JSON.

    For more, use a real templating engine. Kept minimal to avoid surprises.
    """
    import json
    import re

    def resolve(key: str) -> str:
        parts = key.split(".")
        val: Any = context
        for p in parts:
            if isinstance(val, dict):
                val = val.get(p, "")
            else:
                val = getattr(val, p, "")
        if val is None:
            return ""
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}", lambda m: resolve(m.group(1)), template)
