"""search_restaurants node: turns the extracted preferences into a Google Places
Text Search and stores the results in `restaurants`.

Results are kept inside the area being searched, so a rare cuisine returns
nothing rather than restaurants in another country:
- "near me": a box of radius_m around DEFAULT_LOCATION.
- A named place: that place's map box from Google (slightly widened). If Google
  doesn't recognise the name, the search only prefers results near DEFAULT_LOCATION.
"""

from restaurant_agent.config import settings
from restaurant_agent.places import MAX_RADIUS_M, PlacesClient, box_around
from restaurant_agent.state import RestaurantState


def build_query(cuisine: str | None, location: str | None, diet: str | None = None) -> str:
    """('South Indian', 'Indiranagar', 'vegetarian')
    -> 'vegetarian South Indian restaurant in Indiranagar'.
    """
    query = " ".join(part for part in (diet, cuisine, "restaurant") if part)
    return f"{query} in {location}" if location else query


def make_search_restaurants(
    client: PlacesClient,
    default_location: tuple[float, float] | None = settings.default_location,
    radius_m: int = settings.search_radius_m,
):
    """Return the node, bound to `client` (pass a fake in tests)."""

    def search_restaurants(state: RestaurantState) -> dict:
        location = state.get("location")
        if location is None and default_location is None:
            raise ValueError(
                "The request is for somewhere near you, but DEFAULT_LOCATION isn't set. "
                "Add DEFAULT_LOCATION=lat,lng to .env."
            )
        query = build_query(state.get("cuisine"), location, state.get("diet"))
        min_rating = state.get("min_rating")
        if location is None:
            within = box_around(default_location, radius_m)
            restaurants = client.search_text(query, within=within, min_rating=min_rating)
        elif within := client.find_area(location, near=default_location):
            restaurants = client.search_text(query, within=within, min_rating=min_rating)
        else:
            # Unknown place name: keep it in the query and prefer results near home.
            restaurants = client.search_text(
                query, center=default_location, radius_m=MAX_RADIUS_M, min_rating=min_rating
            )
        return {"restaurants": restaurants}

    return search_restaurants
