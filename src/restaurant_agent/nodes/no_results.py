"""End nodes for when there is nothing to recommend. Each explains why, so the
user knows what to change.
"""

from restaurant_agent.state import RestaurantState


def _what_where(state: RestaurantState) -> tuple[str, str]:
    cuisine, location = state.get("cuisine"), state.get("location")
    what = f"{cuisine} restaurants" if cuisine else "restaurants"
    where = f"in {location}" if location else "near you"
    return what, where


def ask_for_location(state: RestaurantState) -> dict:
    """The request is for somewhere near the user, but we don't know where they are."""
    return {
        "recommendations": [],
        "message": "Where should I look? Name an area (e.g. \"dosa in Gunjur, Bangalore\"), "
        "or tell me \"I live in <area>\" and I'll remember it for \"near me\" searches.",
    }


def no_search_results(state: RestaurantState) -> dict:
    """Google returned no restaurants at all."""
    what, where = _what_where(state)
    return {
        "recommendations": [],
        "message": f"Google didn't find any {what} {where}. "
        "Try a broader cuisine or a different area.",
    }


def no_matches(state: RestaurantState) -> dict:
    """Google found restaurants, but the filter removed all of them."""
    limits, tips = [], []
    if state.get("min_rating") is not None:
        limits.append(f"a rating of {state['min_rating']} or more")
        tips.append("lowering the minimum rating")
    if state.get("budget") is not None:
        limits.append(f"a price under ₹{state['budget']} per person")
        tips.append("raising the budget")

    if not limits:
        message = "Google found restaurants, but none of them are currently open for business."
    else:
        message = (
            f"Google found restaurants, but none are open with {' and '.join(limits)}. "
            f"Try {' or '.join(tips)}."
        )
    return {"recommendations": [], "message": message}


def no_relevant_matches(state: RestaurantState) -> dict:
    """Restaurants passed the filter, but the LLM judged none of them a match
    (Google tends to return other restaurants rather than nothing).
    """
    what, where = _what_where(state)
    if not state.get("cuisine"):
        what = "restaurants that match your request"
    return {
        "recommendations": [],
        "message": f"Couldn't find {what} {where}. Google only returned other kinds "
        "of restaurants, so none are shown. Try a broader cuisine or a different area.",
    }
