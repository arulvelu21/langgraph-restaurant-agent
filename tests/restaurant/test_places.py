"""Tests for places.py against a saved response (fixtures/), with no real API calls."""

import json
from pathlib import Path

import httpx
import pytest

from restaurant_agent.places import FIELD_MASK, PlacesClient, PlacesError, box_around

FIXTURE = json.loads((Path(__file__).parent / "fixtures/places_search_text.json").read_text())


def make_client(status=200, payload=FIXTURE):
    """PlacesClient whose HTTP calls return `payload`; also returns the sent requests."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json=payload)

    client = PlacesClient("test-key", http=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, requests


def test_parses_places():
    client, _ = make_client()

    places = client.search_text("South Indian restaurant")

    assert [p["name"] for p in places] == ["Test Dosa House", "Test Meals Mess", "Test Closed Cafe"]
    assert places[0] == {
        "id": "test-place-1",
        "name": "Test Dosa House",
        "address": "1 Test Road, Indiranagar, Bengaluru, Karnataka 560038, India",
        "lat": 12.9784,
        "lng": 77.6408,
        "rating": 4.6,
        "review_count": 3120,
        "price_level": "PRICE_LEVEL_INEXPENSIVE",
        "price_min": 200,
        "price_max": 400,
        "price_currency": "INR",
        "category": "South Indian Restaurant",
        "review_summary": "Diners praise the crisp ghee dosa and filter coffee.",
        "business_status": "OPERATIONAL",
        "maps_url": "https://maps.google.com/?cid=1",
    }


def test_missing_fields_become_none():
    client, _ = make_client()

    no_price, no_rating = client.search_text("x")[1:]

    assert (no_price["price_min"], no_price["price_max"], no_price["price_currency"]) == (None,) * 3
    assert (no_rating["rating"], no_rating["review_count"]) == (None, 0)
    assert no_price["category"] is None
    assert no_price["review_summary"] is None


def test_sends_key_field_mask_and_body():
    client, requests = make_client()

    client.search_text("dosa", center=(12.97, 77.59), radius_m=3000, min_rating=4.2)

    request = requests[0]
    assert request.headers["X-Goog-Api-Key"] == "test-key"
    assert request.headers["X-Goog-FieldMask"] == FIELD_MASK
    assert json.loads(request.content) == {
        "textQuery": "dosa",
        "includedType": "restaurant",
        "pageSize": 20,
        "minRating": 4.0,  # rounded down to a 0.5 step
        "locationBias": {
            "circle": {"center": {"latitude": 12.97, "longitude": 77.59}, "radius": 3000.0}
        },
    }


def test_optional_params_left_out():
    client, requests = make_client()

    client.search_text("dosa")

    body = json.loads(requests[0].content)
    assert "minRating" not in body and "locationBias" not in body


def test_empty_result():
    client, _ = make_client(payload={})
    assert client.search_text("nothing here") == []


def test_api_error_raises_with_google_message():
    client, _ = make_client(
        status=403, payload={"error": {"code": 403, "message": "API key not valid."}}
    )

    with pytest.raises(PlacesError, match="403: API key not valid."):
        client.search_text("dosa")


def test_within_box_is_a_hard_restriction():
    client, requests = make_client()

    client.search_text("dosa", center=(12.97, 77.59), within=((12.9, 77.7), (12.95, 77.75)))

    body = json.loads(requests[0].content)
    assert body["locationRestriction"] == {
        "rectangle": {
            "low": {"latitude": 12.9, "longitude": 77.7},
            "high": {"latitude": 12.95, "longitude": 77.75},
        }
    }
    assert "locationBias" not in body  # the API accepts only one of the two


VIEWPORT = {"places": [{"viewport": {"low": {"latitude": 15.32, "longitude": 76.43},
                                     "high": {"latitude": 15.34, "longitude": 76.47}}}]}


def test_find_area_widens_the_viewport_by_about_2_km():
    client, requests = make_client(payload=VIEWPORT)

    (south, west), (north, east) = client.find_area("Hampi", near=(12.97, 77.59))

    assert south == pytest.approx(15.32 - 0.018, abs=0.001)
    assert north == pytest.approx(15.34 + 0.018, abs=0.001)
    assert west == pytest.approx(76.43 - 0.0186, abs=0.001)  # longer per degree near equator
    assert east == pytest.approx(76.47 + 0.0186, abs=0.001)
    request = requests[0]
    assert request.headers["X-Goog-FieldMask"] == "places.viewport"
    body = json.loads(request.content)
    assert body["textQuery"] == "Hampi" and "includedType" not in body
    assert body["locationBias"]["circle"]["center"] == {"latitude": 12.97, "longitude": 77.59}


def test_find_area_is_cached():
    client, requests = make_client(payload=VIEWPORT)

    assert client.find_area("Hampi") == client.find_area(" hampi ")
    assert len(requests) == 1


def test_find_area_unknown_name():
    client, _ = make_client(payload={})
    assert client.find_area("Nowhereville") is None


def test_box_around():
    (south, west), (north, east) = box_around((12.97, 77.59), 5000)

    assert north - south == pytest.approx(2 * 5 / 111.32)
    assert (west + east) / 2 == pytest.approx(77.59)
    assert east - west > north - south  # degrees of longitude are shorter away from the equator
