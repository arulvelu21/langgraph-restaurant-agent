"""Checkpoints and user preferences survive a fresh Postgres connection.
Needs DATABASE_URL (skipped otherwise).
"""

import os
import uuid

import pytest

from restaurant_agent.graph import build_graph
from restaurant_agent.nodes.preferences import KEY, namespace
from restaurant_agent.persistence import open_persistence
from restaurant_agent.schemas import LLMRanking, Pick, Preferences, UserPreferences
from restaurant_agent.state import Context
from tests.restaurant.test_graph import FakeModel, FakePlaces, place

DATABASE_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")


def test_state_and_preferences_survive_reconnect():
    user = f"test-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": f"{user}:t1"}}
    prefs = Preferences(location="Gunjur", cuisine="dosa", diet="vegetarian",
                        remember=UserPreferences(diet="vegetarian"))
    model = FakeModel([prefs], LLMRanking(picks=[Pick(number=1, reason="r")]))

    with open_persistence(DATABASE_URL) as (checkpointer, store):
        graph = build_graph(model=model, places_client=FakePlaces([place("a")]),
                            checkpointer=checkpointer, store=store)
        graph.invoke({"user_query": "I'm vegetarian, dosa in Gunjur"}, config,
                     context=Context(user_id=user))

    # Fresh connections, like restarting the app.
    with open_persistence(DATABASE_URL) as (checkpointer, store):
        graph = build_graph(model=FakeModel([]), places_client=FakePlaces([]),
                            checkpointer=checkpointer, store=store)
        values = graph.get_state(config).values
        assert values["cuisine"] == "dosa"
        assert values["recommendations"][0]["id"] == "a"
        assert store.get(namespace(user), KEY).value == {"diet": "vegetarian"}
        store.delete(namespace(user), KEY)
