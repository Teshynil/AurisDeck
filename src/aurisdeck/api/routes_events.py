"""Unified multiplexed Server-Sent Events endpoint (`/api/v1/events`)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from aurisdeck.api.deps import get_current_user, get_event_bus
from aurisdeck.api.sse import SSEBroker

router = APIRouter(prefix="/api/v1", dependencies=[Depends(get_current_user)])


def get_sse_broker(request: Request) -> SSEBroker:
    """Return the shared app-level SSEBroker instance, creating it lazily if needed."""
    broker = getattr(request.app.state, "sse_broker", None)
    if broker is None:
        broker = SSEBroker(get_event_bus(request))
        request.app.state.sse_broker = broker
    return broker


@router.get("/events")
async def stream_events(
    request: Request,
    broker: Annotated[SSEBroker, Depends(get_sse_broker)],
) -> StreamingResponse:
    """Stream real-time EventBus events over a single multiplexed SSE connection."""
    return StreamingResponse(
        broker.stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
