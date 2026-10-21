"""Checks conversation memory survives a new connection. Needs DATABASE_URL (skipped otherwise)."""

import os
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph
from tests.test_graph import FakeToolModel

DATABASE_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")


def test_memory_persists_across_connections():
    from langgraph.checkpoint.postgres import PostgresSaver

    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        saver.setup()
        model = FakeToolModel(messages=iter([AIMessage("Nice to meet you!")]))
        build_graph(model=model, checkpointer=saver).invoke(
            {"messages": [HumanMessage("My name is Arul.")]}, config
        )

    # Fresh connection + fresh graph, like restarting the app.
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        model = FakeToolModel(messages=iter([]))
        messages = build_graph(model=model, checkpointer=saver).get_state(config).values[
            "messages"
        ]

    assert [m.content for m in messages] == ["My name is Arul.", "Nice to meet you!"]
