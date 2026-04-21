"""Shared utilities: tokens, JSON repair/extract, retry wrapper."""
from __future__ import annotations

import asyncio
import json
import re

WORDS_PER_TOKEN = 0.75


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def compute_step_budgets(
    target_output_tokens: int,
    num_steps: int,
    think_fraction: float = 0.30,
    min_think_tokens: int = 512,
    max_think_tokens: int = 8_192,
    min_execute_tokens: int = 2_048,
    max_execute_tokens: int = 32_768,
) -> tuple[int, int]:
    """Divide target output budget across plan steps. Returns (think, execute)."""
    num_steps = max(1, num_steps)
    tokens_per_step = max(
        min_think_tokens + min_execute_tokens,
        target_output_tokens // num_steps,
    )
    think_tokens = int(tokens_per_step * think_fraction)
    execute_tokens = tokens_per_step - think_tokens
    think_tokens = max(min_think_tokens, min(think_tokens, max_think_tokens))
    execute_tokens = max(min_execute_tokens, min(execute_tokens, max_execute_tokens))
    return think_tokens, execute_tokens


def truncate_context(
    text: str,
    context_window: int,
    reserved_tokens: int = 4_096,
) -> str:
    """Truncate accumulated context to fit. Keeps the tail (most recent)."""
    if not text:
        return text
    max_words = int((context_window - reserved_tokens) * WORDS_PER_TOKEN)
    words = text.split()
    if len(words) <= max_words:
        return text
    return "... [earlier context truncated] ...\n" + " ".join(words[-max_words:])


def repair_truncated_json(text: str) -> str:
    """Close a JSON array cut off mid-stream."""
    stripped = text.strip()
    if not stripped:
        return text
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    if not stripped.startswith("["):
        return text
    candidate = stripped.rstrip().rstrip(",").rstrip()
    in_string = False
    i = 0
    while i < len(candidate):
        ch = candidate[i]
        if ch == "\\" and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        i += 1
    if in_string:
        candidate += '"'
    candidate += "]"
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass
    return text


def extract_json(response: str):
    """Extract a JSON value from a response with surrounding prose."""
    if not response or response.strip() in ("{}", ""):
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    for pattern in (r"(\[[\s\S]*\])", r"(\{[\s\S]*\})"):
        m = re.search(pattern, response)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    return None


async def stage_retry(fn, max_attempts: int = 4, backoff_base: float = 2.0):
    """Run async fn (returns (result, ok)) up to max_attempts with exp backoff."""
    result = None
    for attempt in range(max_attempts):
        result, ok = await fn()
        if ok:
            return result
        print(f"   [RETRY] Attempt {attempt+1}/{max_attempts} failed ...")
        if attempt < max_attempts - 1:
            wait = backoff_base * (2 ** attempt)
            await asyncio.sleep(wait)
    print(f"   [RETRY] All {max_attempts} attempts exhausted.")
    return result
