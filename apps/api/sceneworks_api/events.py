from __future__ import annotations

import asyncio
import json
from typing import Any


class EventHub:
    def __init__(self) -> None:
        self._subscribers: dict[asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop] = {}

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers[queue] = asyncio.get_running_loop()
        await queue.put({"event": "ready", "data": {"status": "connected"}})
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.pop(queue, None)

    def publish(self, event: str, data: dict[str, Any]) -> None:
        message = {"event": event, "data": data}
        for queue, loop in list(self._subscribers.items()):
            try:
                loop.call_soon_threadsafe(self._put_nowait, queue, message)
            except RuntimeError:
                self.unsubscribe(queue)

    def _put_nowait(self, queue: asyncio.Queue[dict[str, Any]], message: dict[str, Any]) -> None:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            self.unsubscribe(queue)


def encode_sse(message: dict[str, Any]) -> str:
    event = message.get("event", "message")
    data = json.dumps(message.get("data", {}), separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
