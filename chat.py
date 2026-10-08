from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import HumanMessage

from agent import build_agent
from config import Settings
from persistence import async_sqlite_checkpointer


async def ask(
    *,
    thread_id: str,
    query: str,
) -> None:
    settings = Settings.from_env()

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": settings.recursion_limit,
    }

    async with async_sqlite_checkpointer(
        settings.checkpoint_db
    ) as checkpointer:
        graph = build_agent(
            settings,
            checkpointer,
        )

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=query
                    )
                ]
            },
            config=config,
        )

    final_message = result["messages"][-1]
    print(final_message.content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--thread-id",
        required=True,
    )

    parser.add_argument(
        "query",
        nargs="+",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        ask(
            thread_id=args.thread_id,
            query=" ".join(args.query),
        )
    )
