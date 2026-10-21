"""Filtering and scoring. Plain Python with no LLM or network calls, so it is
predictable and easy to unit-test.
"""

import math

# Rough top-of-range ₹ per person for each Google price level, used only when a
# place has no ₹ priceRange. Estimated from Bengaluru results; adjust as needed.
PRICE_LEVEL_MAX_INR = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 200,
    "PRICE_LEVEL_MODERATE": 600,
    "PRICE_LEVEL_EXPENSIVE": 1500,
    "PRICE_LEVEL_VERY_EXPENSIVE": 3000,
}


def max_price(place: dict) -> tuple[int | None, str]:
    """Top of the ₹ per person range and where that number came from:
    "range" (Google's ₹ priceRange), "level" (estimated from priceLevel) or "unknown".
    An open-ended range ("₹2,000+") gives its start, the best number available.
    """
    if place.get("price_currency") in ("INR", None):
        top = place.get("price_max") or place.get("price_min")
        if top is not None:
            return top, "range"
    if place.get("price_level") in PRICE_LEVEL_MAX_INR:
        return PRICE_LEVEL_MAX_INR[place["price_level"]], "level"
    return None, "unknown"


def passes_filters(place: dict, min_rating: float | None, budget: int | None) -> bool:
    """Hard constraints. Unknown price is kept (missing data isn't over budget);
    unknown rating is dropped when a minimum rating was asked for.
    """
    if place.get("business_status") not in ("OPERATIONAL", None):
        return False
    if min_rating is not None and (place.get("rating") is None or place["rating"] < min_rating):
        return False
    if budget is not None:
        price, _ = max_price(place)
        # Top of the range: kept only if a typical meal fits the budget.
        if price is not None and price > budget:
            return False
    return True


def filter_places(places: list[dict], min_rating: float | None, budget: int | None) -> list[dict]:
    """Places that pass the hard constraints, each tagged with price_source."""
    return [
        {**place, "price_source": max_price(place)[1]}
        for place in places
        if passes_filters(place, min_rating, budget)
    ]


# Scoring. Higher is better; roughly on the 0-5 rating scale.
PRIOR_RATING = 4.0  # rating assumed for a place with few reviews
PRIOR_WEIGHT = 200  # reviews needed before a place's own rating counts as much as the prior
UNKNOWN_PRICE_PENALTY = 0.05
DISTANCE_PENALTY_PER_KM = 0.02


def adjusted_rating(rating: float | None, review_count: int) -> float:
    """Rating pulled toward PRIOR_RATING when there are few reviews, so a 4.9
    from 12 reviews doesn't beat a 4.6 from 3,000.
    """
    if rating is None:
        return PRIOR_RATING
    return (review_count * rating + PRIOR_WEIGHT * PRIOR_RATING) / (review_count + PRIOR_WEIGHT)


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Straight-line (haversine) distance between two (lat, lng) points."""
    lat1, lng1, lat2, lng2 = map(math.radians, (*a, *b))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(h))


def score_place(place: dict, origin: tuple[float, float] | None = None) -> dict:
    """The place plus adjusted_rating, distance_km (None without origin) and score."""
    adjusted = adjusted_rating(place.get("rating"), place.get("review_count", 0))
    distance = None
    if origin and place.get("lat") is not None and place.get("lng") is not None:
        distance = distance_km(origin, (place["lat"], place["lng"]))

    score = adjusted
    if max_price(place)[1] == "unknown":
        score -= UNKNOWN_PRICE_PENALTY
    if distance is not None:
        score -= DISTANCE_PENALTY_PER_KM * distance
    return {
        **place,
        "adjusted_rating": round(adjusted, 3),
        "distance_km": None if distance is None else round(distance, 2),
        "score": round(score, 3),
    }


def rank_places(places: list[dict], origin: tuple[float, float] | None = None) -> list[dict]:
    """Score every place and sort best first (ties: more reviews first)."""
    scored = [score_place(p, origin) for p in places]
    return sorted(scored, key=lambda p: (-p["score"], -p.get("review_count", 0)))


def describe_price(place: dict) -> str:
    """Human-readable price, honest about where it came from."""
    price, source = max_price(place)
    if source == "range":
        low, high = place.get("price_min"), place.get("price_max")
        if low is None:
            return f"up to ₹{high}"
        if low <= 1 and high:
            return f"under ₹{high}"
        return f"₹{low}–{high}" if high else f"from ₹{low}"
    if source == "level":
        return f"up to ~₹{price} (estimated)"
    return "price unknown"


def describe_place(place: dict) -> str:
    """One-line summary from the data, e.g. '4.6★ (3,120 reviews) · ₹200–400 · 1.2 km'."""
    parts = []
    if place.get("rating") is not None:
        parts.append(f"{place['rating']}★ ({place.get('review_count', 0):,} reviews)")
    parts.append(describe_price(place))
    if place.get("distance_km") is not None:
        parts.append(f"{place['distance_km']:.1f} km away")
    return " · ".join(parts)
