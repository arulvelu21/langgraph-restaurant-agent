"""Tests for scoring (ranking.py) and the rank_restaurants node."""

import pytest

from restaurant_agent.nodes.rank import make_rank_restaurants
from restaurant_agent.ranking import (
    adjusted_rating,
    describe_place,
    describe_price,
    distance_km,
    rank_places,
)

HOME = (12.9716, 77.5946)


def place(id, rating=4.5, review_count=1000, price_min=200, price_max=400, lat=None, lng=None,
          price_level=None):
    return {
        "id": id,
        "rating": rating,
        "review_count": review_count,
        "price_min": price_min,
        "price_max": price_max,
        "price_currency": "INR" if price_min is not None or price_max is not None else None,
        "price_level": price_level,
        "lat": lat,
        "lng": lng,
    }


def ids(places):
    return [p["id"] for p in places]


def test_many_reviews_beat_few_perfect_ones():
    assert adjusted_rating(4.6, 3000) > adjusted_rating(4.9, 12)


def test_adjusted_rating_without_rating_is_prior():
    assert adjusted_rating(None, 0) == 4.0


def test_distance_km():
    # Bengaluru centre to Indiranagar is about 5 km
    assert distance_km(HOME, (12.9784, 77.6408)) == pytest.approx(5.0, abs=0.2)


def test_higher_score_first():
    places = [
        place("few", rating=4.9, review_count=12),
        place("many", rating=4.6, review_count=3000),
        place("mid", rating=4.4, review_count=1000),
    ]

    assert ids(rank_places(places)) == ["many", "mid", "few"]


def test_unknown_price_ranks_lower():
    ranked = rank_places([place("unknown", price_min=None, price_max=None), place("known")])
    assert ids(ranked) == ["known", "unknown"]


def test_distance_breaks_otherwise_equal_places():
    near = place("near", lat=12.975, lng=77.600)
    far = place("far", lat=13.05, lng=77.70)

    ranked = rank_places([far, near], origin=HOME)

    assert ids(ranked) == ["near", "far"]
    assert ranked[0]["distance_km"] < ranked[1]["distance_km"]


def test_node_uses_distance_only_for_near_me():
    near = place("near", rating=4.4, lat=12.975, lng=77.600)
    far_better = place("far", rating=4.5, lat=13.20, lng=77.70)  # ~25 km away
    node = make_rank_restaurants(default_location=HOME)

    near_me = node({"location": None, "restaurants": [far_better, near]})["restaurants"]
    named = node({"location": "Whitefield", "restaurants": [far_better, near]})["restaurants"]

    assert ids(near_me) == ["near", "far"]
    assert ids(named) == ["far", "near"]
    assert named[0]["distance_km"] is None


def test_node_keeps_shortlist():
    node = make_rank_restaurants(default_location=None, shortlist_size=2)
    restaurants = [place(str(i), rating=4.0 + i / 10) for i in range(5)]

    assert ids(node({"location": None, "restaurants": restaurants})["restaurants"]) == ["4", "3"]


@pytest.mark.parametrize(
    "p, expected",
    [
        (place("a", price_min=200, price_max=400), "₹200–400"),
        (place("a", price_min=1, price_max=200), "under ₹200"),
        (place("a", price_min=None, price_max=None, price_level="PRICE_LEVEL_MODERATE"),
         "up to ~₹600 (estimated)"),
        (place("a", price_min=None, price_max=500), "up to ₹500"),
        (place("a", price_min=None, price_max=None), "price unknown"),
    ],
)
def test_describe_price(p, expected):
    assert describe_price(p) == expected


def test_describe_place():
    p = {**place("a", rating=4.6, review_count=3120), "distance_km": 1.234}
    assert describe_place(p) == "4.6★ (3,120 reviews) · ₹200–400 · 1.2 km away"
