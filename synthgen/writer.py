"""Async buffered JSONL writer — one file open per flush, not per record."""
from __future__ import annotations

import asyncio
import json
from typing import Optional

import aiofiles


class AsyncBufferedWriter:
    """Batches JSONL writes. Flushes when buffer fills or every flush_interval."""

    def __init__(self, filepath: str, flush_interval: float = 1.0,
                 max_buffer: int = 50):
        self._filepath = filepath
        self._buffer: list[str] = []
        self._lock = asyncio.Lock()
        self._flush_interval = flush_interval
        self._max_buffer = max_buffer
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._periodic_flush())

    async def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        async with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._max_buffer:
                await self._do_flush()

    async def _do_flush(self):
        if not self._buffer:
            return
        batch = "".join(self._buffer)
        self._buffer.clear()
        async with aiofiles.open(self._filepath, "a", encoding="utf-8") as f:
            await f.write(batch)
            await f.flush()

    async def _periodic_flush(self):
        while True:
            await asyncio.sleep(self._flush_interval)
            async with self._lock:
                await self._do_flush()

    async def close(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            await self._do_flush()
