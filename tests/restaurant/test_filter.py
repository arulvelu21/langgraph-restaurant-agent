"""Tests for the filter step (ranking.filter_places and the filter_restaurants node)."""

import pytest

from restaurant_agent.nodes.filter import filter_restaurants
from restaurant_agent.ranking import filter_places, max_price


def place(id="p", rating=4.5, price_min=None, price_max=None, price_currency="INR",
          price_level=None, business_status="OPERATIONAL"):
    has_range = price_min is not None or price_max is not None
    return {
        "id": id,
        "rating": rating,
        "price_min": price_min,
        "price_max": price_max,
        "price_currency": price_currency if has_range else None,
        "price_level": price_level,
        "business_status": business_status,
    }


def ids(places):
    return [p["id"] for p in places]


@pytest.mark.parametrize(
    "p, expected",
    [
        (place(price_min=400, price_max=1600, price_level="PRICE_LEVEL_MODERATE"),
         (1600, "range")),
        (place(price_min=2000), (2000, "range")),  # open-ended "₹2,000+"
        (place(price_level="PRICE_LEVEL_MODERATE"), (600, "level")),
        (place(), (None, "unknown")),
        (place(price_min=15, price_max=30, price_currency="USD",
               price_level="PRICE_LEVEL_EXPENSIVE"), (1500, "level")),
    ],
)
def test_max_price(p, expected):
    assert max_price(p) == expected


def test_rating_is_inclusive_and_missing_rating_dropped():
    places = [place("a", rating=4.2), place("b", rating=4.19), place("c", rating=None)]

    assert ids(filter_places(places, min_rating=4.2, budget=None)) == ["a"]


def test_budget_uses_top_of_range():
    places = [
        place("cheap", price_min=1, price_max=200),
        place("edge", price_min=400, price_max=800),
        place("wide", price_min=200, price_max=1400),  # can't count on eating for ₹800
        place("pricey", price_min=1000, price_max=1500),
    ]

    assert ids(filter_places(places, min_rating=None, budget=800)) == ["cheap", "edge"]


def test_budget_falls_back_to_price_level_and_keeps_unknown():
    places = [
        place("moderate", price_level="PRICE_LEVEL_MODERATE"),  # up to ~₹600
        place("expensive", price_level="PRICE_LEVEL_EXPENSIVE"),  # up to ~₹1500
        place("unknown"),
    ]

    kept = filter_places(places, min_rating=None, budget=800)

    assert ids(kept) == ["moderate", "unknown"]
    assert [p["price_source"] for p in kept] == ["level", "unknown"]


def test_closed_places_dropped():
    places = [
        place("open"),
        place("temp", business_status="CLOSED_TEMPORARILY"),
        place("gone", business_status="CLOSED_PERMANENTLY"),
        place("no-status", business_status=None),
    ]

    assert ids(filter_places(places, min_rating=None, budget=None)) == ["open", "no-status"]


def test_no_constraints_keeps_everything_open():
    places = [place("a", rating=None), place("b", price_min=5000, price_max=8000)]

    assert ids(filter_places(places, min_rating=None, budget=None)) == ["a", "b"]


def test_node_replaces_restaurants():
    state = {
        "min_rating": 4.2,
        "budget": 800,
        "restaurants": [place("keep", rating=4.4, price_min=200, price_max=400),
                        place("drop", rating=3.9)],
    }

    update = filter_restaurants(state)

    assert ids(update["restaurants"]) == ["keep"]
    assert update["restaurants"][0]["price_source"] == "range"
