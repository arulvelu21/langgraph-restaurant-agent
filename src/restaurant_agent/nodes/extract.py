"""extract_preferences node: the LLM reads user_query and fills in the search
preferences (location, cuisine, budget, min_rating, diet), plus any standing
preferences the user asked to be remembered. Anything not stated is None.

When the thread already has a search (kept by the checkpointer), it is shown to
the LLM so a follow-up like "something cheaper" changes that search.
"""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from restaurant_agent.prompts import EXTRACT_PROMPT
from restaurant_agent.schemas import Preferences
from restaurant_agent.state import RestaurantState

SEARCH_FIELDS = ("location", "cuisine", "budget", "min_rating", "diet")


def current_query(state: RestaurantState) -> str:
    """The request to handle: a new user message (chat UIs end each turn with an
    answer, so a trailing user message is new), else user_query.
    """
    messages = state.get("messages") or []
    if messages and messages[-1].type == "human":
        return messages[-1].text
    return state["user_query"]


def build_extract_input(query: str, state: RestaurantState) -> str:
    previous = {field: state.get(field) for field in SEARCH_FIELDS}
    if not any(value is not None for value in previous.values()):
        return query
    return (
        f"Previous search: {json.dumps(previous, ensure_ascii=False)}\n\n"
        f"New message: {query}"
    )


def make_extract_preferences(model: BaseChatModel):
    """Return the node, bound to `model` (pass a fake in tests)."""
    # nostream: keep this internal JSON out of chat UIs that stream LLM tokens
    extractor = model.with_structured_output(Preferences).with_config(tags=["nostream"])

    def extract_preferences(state: RestaurantState) -> dict:
        query = current_query(state)
        prefs = extractor.invoke(
            [SystemMessage(EXTRACT_PROMPT), HumanMessage(build_extract_input(query, state))]
        )
        return {
            "user_query": query,
            **prefs.model_dump(include=set(SEARCH_FIELDS)),
            "new_preferences": prefs.remember.model_dump(exclude_none=True),
            "message": None,  # clear a "no results" message left from the previous request
        }

    return extract_preferences
