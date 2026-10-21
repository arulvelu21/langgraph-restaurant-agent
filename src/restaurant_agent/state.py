"""The graph's state, filled in step by step as the request moves through the pipeline.

    messages          the chat: user requests in, the agent's answers out (for chat UIs)
    user_query        the request being handled: the latest user message, or set
                      directly by the caller
    location..diet    set by the extract step, then gaps filled from saved user
                      preferences; None when neither says
    new_preferences   standing preferences the user stated this turn (to save)
    user_preferences  the user's saved preferences, after saving the new ones
    restaurants       search: Google's results -> filter: those that pass ->
                      rank: the shortlist for the LLM
    recommendations   set by the LLM rank step (top 3, each with a reason)
    message           set instead of recommendations when nothing was found, saying why

With a checkpointer, the state of a thread is kept between requests, so the
extract step can treat a follow-up ("something cheaper") as a change to the
previous search.
"""

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class RestaurantState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str

    # Search requirements
    location: str | None  # None = near the user (saved home, else DEFAULT_LOCATION)
    cuisine: str | None
    budget: int | None  # max ₹ per person
    min_rating: float | None
    diet: str | None  # e.g. "vegetarian"

    new_preferences: dict
    user_preferences: dict

    restaurants: list[dict]
    recommendations: list[dict]
    message: str | None


@dataclass
class Context:
    """Per-run settings passed as graph.invoke(..., context=Context(user_id=...))."""

    user_id: str = "default"
