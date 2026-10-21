"""Terminal chat with the restaurant finder.

    restaurant --user arul --thread dinner

--thread keeps a conversation: reuse it to continue, so follow-ups like
"something cheaper" change the last search. --user keeps preferences
("remember I'm vegetarian") across all of that user's threads.
"""

import argparse

from langchain_core.messages import HumanMessage

from restaurant_agent.config import settings
from restaurant_agent.graph import build_graph
from restaurant_agent.nodes.preferences import KEY, namespace
from restaurant_agent.nodes.respond import format_preferences
from restaurant_agent.persistence import open_persistence
from restaurant_agent.state import Context

HELP = "Commands: /prefs shows saved preferences, /forget clears them, quit exits."


def main() -> None:
    parser = argparse.ArgumentParser(description="Find restaurants in plain language.")
    parser.add_argument("--user", default="default", help="whose preferences to use and save")
    parser.add_argument("--thread", default="default", help="conversation id; reuse to continue")
    args = parser.parse_args()

    with open_persistence() as (checkpointer, store):
        graph = build_graph(checkpointer=checkpointer, store=store)
        config = {"configurable": {"thread_id": f"{args.user}:{args.thread}"}}
        context = Context(user_id=args.user)

        storage = "postgres" if settings.database_url else "in-memory"
        saved = store.get(namespace(args.user), KEY)
        print(f"Restaurant finder ({settings.provider}/{settings.model}, {storage} storage).")
        print(f"User '{args.user}' · thread '{args.thread}'")
        print(f"Saved preferences: {format_preferences(saved.value if saved else {})}")
        print(HELP)

        while True:
            try:
                query = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in {"quit", "exit"}:
                break
            if query == "/prefs":
                saved = store.get(namespace(args.user), KEY)
                print(format_preferences(saved.value if saved else {}))
                continue
            if query == "/forget":
                store.delete(namespace(args.user), KEY)
                print("Saved preferences cleared.")
                continue
            if not query:
                continue
            result = graph.invoke({"messages": [HumanMessage(query)]}, config, context=context)
            print(f"\nAgent:\n{result['messages'][-1].text}")


if __name__ == "__main__":
    main()
