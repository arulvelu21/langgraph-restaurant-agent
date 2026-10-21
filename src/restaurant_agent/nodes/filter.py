"""filter_restaurants node: drops restaurants that break the user's hard
constraints (closed, below min rating, over budget). Replaces `restaurants`
with the ones that pass.
"""

from restaurant_agent.ranking import filter_places
from restaurant_agent.state import RestaurantState


def filter_restaurants(state: RestaurantState) -> dict:
    return {
        "restaurants": filter_places(
            state.get("restaurants", []), state.get("min_rating"), state.get("budget")
        )
    }
