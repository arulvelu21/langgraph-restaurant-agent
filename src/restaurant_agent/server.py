"""The graph served by `langgraph dev` (see langgraph.json), e.g. for Agent Chat
UI or Studio. The server supplies its own checkpointer and store, so none are
passed here. Requests without a user_id context use the "default" user.
"""

from restaurant_agent.graph import build_graph

graph = build_graph()
