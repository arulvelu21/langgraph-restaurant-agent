"""Interactive terminal chat with the agent."""

import argparse
from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer

from agent.config import settings
from agent.graph import build_graph


@contextmanager
def open_checkpointer() -> Iterator[Checkpointer]:
    """Postgres when DATABASE_URL is set, otherwise in-memory."""
    if not settings.database_url:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(settings.database_url) as saver:
        saver.setup()  # creates/migrates checkpoint tables; safe to run every time
        yield saver


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the LangGraph agent.")
    parser.add_argument(
        "--thread", default="default", help="conversation id; reuse it to resume"
    )
    args = parser.parse_args()

    with open_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": args.thread}}

        storage = "postgres" if settings.database_url else "in-memory"
        history = graph.get_state(config).values.get("messages", [])
        print(f"LangGraph agent ready ({settings.provider}/{settings.model}).")
        print(f"Thread '{args.thread}' · {storage} memory · {len(history)} prior messages")
        print("Type 'quit' to exit.")

        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in {"quit", "exit"}:
                break
            if not user:
                continue
            result = graph.invoke({"messages": [HumanMessage(user)]}, config)
            print(f"\nAgent: {result['messages'][-1].text}")


if __name__ == "__main__":
    main()
