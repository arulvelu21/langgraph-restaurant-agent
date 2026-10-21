"""Google Places API (New) client. The only module that talks to Google."""

import math

import httpx

SEARCH_TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

# Only these fields are requested, and Google bills by the fields requested.
FIELD_MASK = ",".join(
    f"places.{field}"
    for field in [
        "id",
        "displayName",
        "formattedAddress",
        "location",
        "rating",
        "userRatingCount",
        "priceLevel",
        "priceRange",
        "primaryTypeDisplayName",
        "reviewSummary",
        "businessStatus",
        "googleMapsUri",
    ]
)

MAX_RADIUS_M = 50_000  # largest locationBias radius the API accepts
AREA_PADDING_KM = 2  # widen an area's map box so places just outside it still count
KM_PER_DEGREE_LAT = 111.32

LatLng = tuple[float, float]
Box = tuple[LatLng, LatLng]  # (south-west corner, north-east corner)


class PlacesError(Exception):
    """The Places API returned an error (bad key, API not enabled, bad request...)."""


class PlacesClient:
    def __init__(self, api_key: str, http: httpx.Client | None = None):
        self.api_key = api_key
        self.http = http or httpx.Client(timeout=10)
        self._areas: dict = {}  # find_area cache

    def search_text(
        self,
        query: str,
        *,
        center: LatLng | None = None,
        radius_m: int = 5000,
        within: Box | None = None,
        min_rating: float | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Run a Text Search and return the places as plain dicts (see parse_place).

        `within` keeps results inside a map box (a hard limit); otherwise `center`
        and `radius_m` only make nearby results more likely.
        """
        body: dict = {"textQuery": query, "includedType": "restaurant", "pageSize": max_results}
        if within:
            (south, west), (north, east) = within
            body["locationRestriction"] = {
                "rectangle": {
                    "low": {"latitude": south, "longitude": west},
                    "high": {"latitude": north, "longitude": east},
                }
            }
        elif center:
            lat, lng = center
            body["locationBias"] = {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(min(radius_m, MAX_RADIUS_M)),
                }
            }
        if min_rating:
            # The API only accepts steps of 0.5, so round down here; the filter
            # step applies the exact rating.
            body["minRating"] = math.floor(min_rating * 2) / 2

        data = self._post(body, FIELD_MASK)
        return [parse_place(p) for p in data.get("places", [])]

    def find_area(self, name: str, near: LatLng | None = None) -> Box | None:
        """The map box of a named area ("Gunjur, Bangalore"), widened by
        AREA_PADDING_KM, or None if Google doesn't know it. Cached per name.
        """
        key = (name.strip().lower(), near)
        if key not in self._areas:
            body: dict = {"textQuery": name, "pageSize": 1}
            if near:
                body["locationBias"] = {
                    "circle": {
                        "center": {"latitude": near[0], "longitude": near[1]},
                        "radius": float(MAX_RADIUS_M),
                    }
                }
            places = self._post(body, "places.viewport").get("places", [])
            viewport = places[0].get("viewport") if places else None
            self._areas[key] = _padded_box(viewport) if viewport else None
        return self._areas[key]

    def _post(self, body: dict, field_mask: str) -> dict:
        resp = self.http.post(
            SEARCH_TEXT_URL,
            json=body,
            headers={"X-Goog-Api-Key": self.api_key, "X-Goog-FieldMask": field_mask},
        )
        if resp.is_error:
            raise PlacesError(f"Places API error {resp.status_code}: {_error_message(resp)}")
        return resp.json()


def box_around(center: LatLng, radius_m: int) -> Box:
    """Square map box reaching radius_m from center in each direction."""
    lat, lng = center
    dlat = radius_m / 1000 / KM_PER_DEGREE_LAT
    dlng = radius_m / 1000 / (KM_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return (lat - dlat, lng - dlng), (lat + dlat, lng + dlng)



def _padded_box(viewport: dict) -> Box:
    low, high = viewport["low"], viewport["high"]
    mid_lat = (low["latitude"] + high["latitude"]) / 2
    dlat = AREA_PADDING_KM / KM_PER_DEGREE_LAT
    dlng = AREA_PADDING_KM / (KM_PER_DEGREE_LAT * math.cos(math.radians(mid_lat)))
    return (
        (low["latitude"] - dlat, low["longitude"] - dlng),
        (high["latitude"] + dlat, high["longitude"] + dlng),
    )


def parse_place(place: dict) -> dict:
    """Flatten one API place into the fields the rest of the agent uses."""
    location = place.get("location", {})
    price_range = place.get("priceRange", {})
    start, end = price_range.get("startPrice"), price_range.get("endPrice")
    return {
        "id": place["id"],
        "name": place.get("displayName", {}).get("text"),
        "address": place.get("formattedAddress"),
        "lat": location.get("latitude"),
        "lng": location.get("longitude"),
        "rating": place.get("rating"),
        "review_count": place.get("userRatingCount", 0),
        "price_level": place.get("priceLevel"),  # e.g. "PRICE_LEVEL_MODERATE"
        # priceRange is per person and often missing; None when Google doesn't have it
        "price_min": _money_units(start),
        "price_max": _money_units(end),
        "price_currency": (start or end or {}).get("currencyCode"),
        "category": place.get("primaryTypeDisplayName", {}).get("text"),  # "South Indian Restaurant"
        # Google's AI summary of the reviews; often missing outside big cities
        "review_summary": place.get("reviewSummary", {}).get("text", {}).get("text"),
        "business_status": place.get("businessStatus"),  # e.g. "OPERATIONAL"
        "maps_url": place.get("googleMapsUri"),
    }


def _money_units(money: dict | None) -> int | None:
    if not money or "units" not in money:
        return None
    return int(money["units"])  # sent as a string, e.g. "400"


def _error_message(resp: httpx.Response) -> str:
    try:
        return resp.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return resp.text[:200]
