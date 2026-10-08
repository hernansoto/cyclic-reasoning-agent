from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from collections.abc import AsyncIterator, Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def async_sqlite_checkpointer(
    database_path: Path,
) -> AsyncIterator[AsyncSqliteSaver]:
    """
    Checkpointer SQLite para el agente asíncrono.

    LangGraph separa actualmente `SqliteSaver` (sincrónico) de
    `AsyncSqliteSaver`. Como el entregable usa `ainvoke`/`astream`, la variante
    async evita bloquear y soporta los métodos async del grafo.
    """
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    async with AsyncSqliteSaver.from_conn_string(
        str(database_path)
    ) as saver:
        yield saver


@contextmanager
def sync_sqlite_checkpointer(
    database_path: Path,
) -> Iterator[SqliteSaver]:
    """
    Variante SqliteSaver sincrónica incluida como referencia y para inspección.

    El runtime principal del proyecto usa AsyncSqliteSaver.
    """
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(database_path),
        check_same_thread=False,
    )

    try:
        yield SqliteSaver(connection)
    finally:
        connection.close()
