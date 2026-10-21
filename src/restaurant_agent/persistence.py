"""Checkpointer (per-thread state) and store (per-user preferences).

Postgres when DATABASE_URL is set (start it with: docker compose up -d),
otherwise in-memory, which is lost when the program exits.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Checkpointer

from restaurant_agent.config import settings


@contextmanager
def open_persistence(
    database_url: str | None = settings.database_url,
) -> Iterator[tuple[Checkpointer, BaseStore]]:
    if not database_url:
        yield InMemorySaver(), InMemoryStore()
        return

    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    with (
        PostgresSaver.from_conn_string(database_url) as checkpointer,
        PostgresStore.from_conn_string(database_url) as store,
    ):
        # Create/migrate the tables; safe to run every time.
        checkpointer.setup()
        store.setup()
        yield checkpointer, store
