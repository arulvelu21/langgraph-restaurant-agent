"""Tests for apply_user_preferences: saving and applying per-user preferences."""

from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from restaurant_agent.nodes.preferences import KEY, apply_user_preferences, namespace
from restaurant_agent.state import Context

EMPTY_SEARCH = {"location": None, "cuisine": "dosa", "budget": None, "min_rating": None,
                "diet": None, "new_preferences": {}}


def run(state, store, user="arul"):
    return apply_user_preferences(state, Runtime(context=Context(user_id=user), store=store))


def saved(store, user="arul"):
    item = store.get(namespace(user), KEY)
    return item.value if item else None


def test_saves_new_preferences_and_merges_with_old():
    store = InMemoryStore()
    store.put(namespace("arul"), KEY, {"diet": "vegetarian", "budget": 500})

    update = run({**EMPTY_SEARCH, "new_preferences": {"budget": 600, "home_location": "Gunjur"}},
                 store)

    expected = {"diet": "vegetarian", "budget": 600, "home_location": "Gunjur"}
    assert saved(store) == expected
    assert update["user_preferences"] == expected


def test_fills_empty_fields_from_saved_preferences():
    store = InMemoryStore()
    store.put(namespace("arul"), KEY, {"diet": "vegetarian", "budget": 500, "min_rating": 4.0,
                                       "home_location": "Gunjur, Bengaluru"})

    update = run(EMPTY_SEARCH, store)

    assert {k: update[k] for k in ("location", "diet", "budget", "min_rating")} == {
        "location": "Gunjur, Bengaluru",
        "diet": "vegetarian",
        "budget": 500,
        "min_rating": 4.0,
    }


def test_request_values_win_over_saved():
    store = InMemoryStore()
    store.put(namespace("arul"), KEY, {"budget": 500, "home_location": "Gunjur"})

    update = run({**EMPTY_SEARCH, "location": "Koramangala", "budget": 1200}, store)

    assert "location" not in update and "budget" not in update


def test_preferences_are_per_user():
    store = InMemoryStore()
    run({**EMPTY_SEARCH, "new_preferences": {"diet": "vegan"}}, store, user="arul")

    update = run(EMPTY_SEARCH, store, user="priya")

    assert saved(store, "priya") is None
    assert "diet" not in update


def test_without_store_does_nothing():
    update = run({**EMPTY_SEARCH, "new_preferences": {"diet": "vegan"}}, store=None)

    assert update == {"user_preferences": {}}
