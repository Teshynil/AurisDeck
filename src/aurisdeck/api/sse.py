"""Server-Sent Events (SSE) broker multiplexing EventBus envelopes with heartbeats."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Request

from aurisdeck.workers.events import EventBus


def format_sse_event(seq: int, event_type: str, data: dict[str, Any]) -> str:
    """Format an SSE frame with sequential id, event name, and single-line JSON data."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {seq}\nevent: {event_type}\ndata: {payload}\n\n"


class SSEBroker:
    """Multiplexed SSE stream publisher backed by AurisDeck's internal EventBus."""

    def __init__(self, event_bus: EventBus, heartbeat_seconds: float = 15.0) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")
        self._event_bus = event_bus
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._seq_counter = itertools.count(1)

    @property
    def heartbeat_seconds(self) -> float:
        return self._heartbeat_seconds

    @staticmethod
    def format_sse_event(seq: int, event_type: str, data: dict[str, Any]) -> str:
        """Format an SSE frame with sequential id, event name, and single-line JSON data."""
        return format_sse_event(seq, event_type, data)

    async def stream(self, request: Request | None = None) -> AsyncIterator[str]:
        """Yield formatted SSE events and periodic keepalive comments until disconnect."""
        queue = self._event_bus.subscribe()
        try:
            while True:
                if request is not None and await request.is_disconnected():
                    break
                try:
                    envelope = await asyncio.wait_for(
                        queue.get(),
                        timeout=self._heartbeat_seconds,
                    )
                except TimeoutError:
                    if request is not None and await request.is_disconnected():
                        break
                    yield ": keepalive\n\n"
                    continue

                seq = next(self._seq_counter)
                event_type = str(envelope.get("type") or "message")
                yield self.format_sse_event(seq, event_type, envelope)
        finally:
            self._event_bus.unsubscribe(queue)
