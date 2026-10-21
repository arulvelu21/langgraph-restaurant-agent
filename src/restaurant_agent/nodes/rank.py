"""rank_restaurants node: scores the filtered restaurants (adjusted rating,
distance, missing price) and keeps the best few as a shortlist for the LLM.
"""

from restaurant_agent.config import settings
from restaurant_agent.ranking import rank_places
from restaurant_agent.state import RestaurantState

SHORTLIST_SIZE = 8


def make_rank_restaurants(
    default_location: tuple[float, float] | None = settings.default_location,
    shortlist_size: int = SHORTLIST_SIZE,
):
    def rank_restaurants(state: RestaurantState) -> dict:
        # Distance only matters for "near me"; for a named place, DEFAULT_LOCATION
        # may be nowhere near where the user wants to eat.
        origin = default_location if state.get("location") is None else None
        ranked = rank_places(state.get("restaurants", []), origin)
        return {"restaurants": ranked[:shortlist_size]}

    return rank_restaurants
