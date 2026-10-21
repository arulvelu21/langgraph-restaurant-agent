"""Tests for the search_restaurants node, using a fake Places client."""

import pytest

from restaurant_agent.nodes.search import build_query, make_search_restaurants
from restaurant_agent.places import MAX_RADIUS_M, box_around

HOME = (12.97, 77.59)
INDIRANAGAR_BOX = ((12.96, 77.63), (12.99, 77.66))


class FakeClient:
    def __init__(self, results=(), areas=None):
        self.results = list(results)
        self.areas = areas or {}
        self.calls = []
        self.area_lookups = []

    def search_text(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.results

    def find_area(self, name, near=None):
        self.area_lookups.append((name, near))
        return self.areas.get(name)


@pytest.mark.parametrize(
    "cuisine, location, diet, expected",
    [
        ("South Indian", "Indiranagar", None, "South Indian restaurant in Indiranagar"),
        ("South Indian", None, None, "South Indian restaurant"),
        (None, "Koramangala", None, "restaurant in Koramangala"),
        (None, None, None, "restaurant"),
        ("dosa", "Indiranagar", "vegetarian", "vegetarian dosa restaurant in Indiranagar"),
    ],
)
def test_build_query(cuisine, location, diet, expected):
    assert build_query(cuisine, location, diet) == expected


def test_near_me_stays_within_radius_of_default_location():
    client = FakeClient(results=[{"id": "a"}])
    node = make_search_restaurants(client, default_location=HOME, radius_m=3000)

    update = node({"location": None, "cuisine": "South Indian", "min_rating": 4.2})

    assert update == {"restaurants": [{"id": "a"}]}
    assert client.calls == [
        ("South Indian restaurant", {"within": box_around(HOME, 3000), "min_rating": 4.2})
    ]
    assert client.area_lookups == []


def test_named_location_stays_within_that_area():
    client = FakeClient(areas={"Indiranagar": INDIRANAGAR_BOX})
    node = make_search_restaurants(client, default_location=HOME)

    node({"location": "Indiranagar", "cuisine": "dosa", "min_rating": None})

    assert client.area_lookups == [("Indiranagar", HOME)]
    assert client.calls == [("dosa restaurant in Indiranagar", {"within": INDIRANAGAR_BOX, "min_rating": None})]


def test_unknown_area_falls_back_to_preferring_results_near_home():
    client = FakeClient()
    node = make_search_restaurants(client, default_location=HOME)

    node({"location": "Nowhereville", "cuisine": "dosa", "min_rating": None})

    assert client.calls[0][1] == {"center": HOME, "radius_m": MAX_RADIUS_M, "min_rating": None}


def test_named_location_works_without_default_location():
    client = FakeClient(areas={"Indiranagar": INDIRANAGAR_BOX})
    node = make_search_restaurants(client, default_location=None)

    node({"location": "Indiranagar", "cuisine": None, "min_rating": None})

    assert client.area_lookups == [("Indiranagar", None)]
    assert client.calls[0][1]["within"] == INDIRANAGAR_BOX


def test_near_me_without_default_location_fails_clearly():
    node = make_search_restaurants(FakeClient(), default_location=None)

    with pytest.raises(ValueError, match="DEFAULT_LOCATION"):
        node({"location": None, "cuisine": "dosa", "min_rating": None})
