from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    """Serializa mensajes observables sin incluir razonamiento interno."""
    base: dict[str, Any] = {
        "type": message.type,
        "content": message.content,
    }

    if isinstance(message, HumanMessage):
        base["role"] = "user"

    if isinstance(message, AIMessage):
        base["role"] = "assistant"
        base["tool_calls"] = message.tool_calls

    if isinstance(message, ToolMessage):
        base["role"] = "tool"
        base["tool_call_id"] = message.tool_call_id
        base["tool_name"] = message.name

    return base


def event_to_trace_entries(
    event: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for node_name, node_update in event.items():
        if not isinstance(node_update, dict):
            continue

        messages = node_update.get("messages", [])

        for message in messages:
            entries.append(
                {
                    "node": node_name,
                    "message": message_to_dict(message),
                }
            )

    return entries


def write_trace(
    entries: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            entries,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
