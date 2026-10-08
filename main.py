from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langchain_core.messages import HumanMessage

from agent import build_agent
from config import Settings
from persistence import async_sqlite_checkpointer
from trace_utils import event_to_trace_entries, write_trace


async def run_turn(
    graph,
    *,
    thread_id: str,
    user_message: str,
    recursion_limit: int,
    trace_entries: list[dict],
) -> str:
    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": recursion_limit,
    }

    final_text = ""

    async for event in graph.astream(
        {
            "messages": [
                HumanMessage(
                    content=user_message
                )
            ]
        },
        config=config,
        stream_mode="updates",
    ):
        trace_entries.extend(
            event_to_trace_entries(event)
        )

        for update in event.values():
            if not isinstance(update, dict):
                continue

            for message in update.get(
                "messages",
                [],
            ):
                if (
                    getattr(message, "type", "") == "ai"
                    and not getattr(
                        message,
                        "tool_calls",
                        [],
                    )
                    and isinstance(
                        message.content,
                        str,
                    )
                    and message.content.strip()
                ):
                    final_text = message.content

    return final_text


async def demo(
    thread_id: str,
    trace_path: Path,
) -> None:
    settings = Settings.from_env()

    trace_entries: list[dict] = []

    async with async_sqlite_checkpointer(
        settings.checkpoint_db
    ) as checkpointer:
        graph = build_agent(
            settings,
            checkpointer,
        )

        first_question = (
            "¿Cuántos pedidos tuvo el cliente 102, cuál fue el importe "
            "total de todos sus pedidos y qué producto compró en el último?"
        )

        print("\n=== TURNO 1: MULTI-PASO ===")
        print(f"thread_id: {thread_id}")
        print(f"Usuario: {first_question}\n")

        first_answer = await run_turn(
            graph,
            thread_id=thread_id,
            user_message=first_question,
            recursion_limit=settings.recursion_limit,
            trace_entries=trace_entries,
        )

        print(f"Agente: {first_answer}")

        second_question = (
            "¿Y en qué fecha fue ese último pedido?"
        )

        print("\n=== TURNO 2: MEMORIA MISMO thread_id ===")
        print(f"Usuario: {second_question}\n")

        second_answer = await run_turn(
            graph,
            thread_id=thread_id,
            user_message=second_question,
            recursion_limit=settings.recursion_limit,
            trace_entries=trace_entries,
        )

        print(f"Agente: {second_answer}")

    write_trace(
        trace_entries,
        trace_path,
    )

    print(
        f"\nTraza observable guardada en: "
        f"{trace_path.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agente cíclico LangGraph con memoria SQLite."
        )
    )

    parser.add_argument(
        "--thread-id",
        default="cliente-102-demo",
        help=(
            "Identificador persistente de conversación."
        ),
    )

    parser.add_argument(
        "--trace",
        default="./traces/latest_trace.json",
        help="Ruta para guardar la traza observable.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        demo(
            thread_id=args.thread_id,
            trace_path=Path(args.trace),
        )
    )
