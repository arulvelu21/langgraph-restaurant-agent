"""Builds and compiles the finder graph.

    START -> extract_preferences -> apply_user_preferences
        -> (near me, location unknown) ask_for_location -> END
        -> search_restaurants
        -> (none found)  no_search_results -> END
        -> filter_restaurants
            -> (none left) no_matches -> END
            -> rank_restaurants -> llm_rank_restaurants
                -> (none relevant) no_relevant_matches -> END
                -> END

Every end above goes through `respond`, which adds the answer to `messages`.

Pass a checkpointer to keep each thread's state between requests (follow-ups),
and a store to keep each user's preferences across threads.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from restaurant_agent.config import settings
from restaurant_agent.nodes.extract import make_extract_preferences
from restaurant_agent.nodes.filter import filter_restaurants
from restaurant_agent.nodes.llm_rank import make_llm_rank_restaurants
from restaurant_agent.nodes.no_results import (
    ask_for_location,
    no_matches,
    no_relevant_matches,
    no_search_results,
)
from restaurant_agent.nodes.preferences import apply_user_preferences
from restaurant_agent.nodes.rank import make_rank_restaurants
from restaurant_agent.nodes.respond import respond
from restaurant_agent.nodes.search import make_search_restaurants
from restaurant_agent.places import PlacesClient
from restaurant_agent.state import Context, RestaurantState


def make_route_after_preferences(default_location: tuple[float, float] | None):
    def route_after_preferences(
        state: RestaurantState,
    ) -> Literal["search_restaurants", "ask_for_location"]:
        if state.get("location") is None and default_location is None:
            return "ask_for_location"
        return "search_restaurants"

    return route_after_preferences


def route_after_search(state: RestaurantState) -> Literal["filter_restaurants", "no_search_results"]:
    return "filter_restaurants" if state.get("restaurants") else "no_search_results"


def route_after_filter(state: RestaurantState) -> Literal["rank_restaurants", "no_matches"]:
    return "rank_restaurants" if state.get("restaurants") else "no_matches"


def route_after_llm_rank(state: RestaurantState) -> Literal["respond", "no_relevant_matches"]:
    return "respond" if state.get("recommendations") else "no_relevant_matches"


def build_graph(
    model: BaseChatModel | None = None,
    places_client: PlacesClient | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    default_location: tuple[float, float] | None = settings.default_location,
):
    """Compile the graph. Real model and Places client by default; pass fakes in tests."""
    if model is None:
        from restaurant_agent.llm import get_model

        model = get_model()
    if places_client is None:
        if not settings.google_places_api_key:
            raise ValueError("GOOGLE_PLACES_API_KEY isn't set. Add it to .env.")
        places_client = PlacesClient(settings.google_places_api_key)

    builder = StateGraph(RestaurantState, context_schema=Context)
    builder.add_node("extract_preferences", make_extract_preferences(model))
    builder.add_node("apply_user_preferences", apply_user_preferences)
    builder.add_node("ask_for_location", ask_for_location)
    builder.add_node(
        "search_restaurants", make_search_restaurants(places_client, default_location)
    )
    builder.add_node("filter_restaurants", filter_restaurants)
    builder.add_node("rank_restaurants", make_rank_restaurants())
    builder.add_node("llm_rank_restaurants", make_llm_rank_restaurants(model))
    builder.add_node("no_search_results", no_search_results)
    builder.add_node("no_matches", no_matches)
    builder.add_node("no_relevant_matches", no_relevant_matches)
    builder.add_node("respond", respond)

    builder.add_edge(START, "extract_preferences")
    builder.add_edge("extract_preferences", "apply_user_preferences")
    builder.add_conditional_edges(
        "apply_user_preferences", make_route_after_preferences(default_location)
    )
    builder.add_conditional_edges("search_restaurants", route_after_search)
    builder.add_conditional_edges("filter_restaurants", route_after_filter)
    builder.add_edge("rank_restaurants", "llm_rank_restaurants")
    builder.add_conditional_edges("llm_rank_restaurants", route_after_llm_rank)
    for node in ("ask_for_location", "no_search_results", "no_matches", "no_relevant_matches"):
        builder.add_edge(node, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer, store=store)
