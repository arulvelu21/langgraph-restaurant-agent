"""Tests for the extract_preferences node, using a fake model (no API calls)."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from restaurant_agent.nodes.extract import current_query, make_extract_preferences
from restaurant_agent.prompts import EXTRACT_PROMPT
from restaurant_agent.schemas import Preferences, UserPreferences


class FakeModel:
    """Returns a fixed Preferences and records the messages it was sent."""

    def __init__(self, result: Preferences):
        self.result = result
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is Preferences

        def respond(messages):
            self.calls.append(messages)
            return self.result

        return RunnableLambda(respond)


def test_returns_extracted_preferences():
    model = FakeModel(
        Preferences(location="Indiranagar", cuisine="South Indian", budget=800, min_rating=4.2)
    )
    node = make_extract_preferences(model)

    update = node({"user_query": "South Indian in Indiranagar, rating > 4.2, under ₹800/person"})

    assert update == {
        "user_query": "South Indian in Indiranagar, rating > 4.2, under ₹800/person",
        "location": "Indiranagar",
        "cuisine": "South Indian",
        "budget": 800,
        "min_rating": 4.2,
        "diet": None,
        "new_preferences": {},
        "message": None,
    }


def test_unstated_fields_are_none():
    node = make_extract_preferences(FakeModel(Preferences(cuisine="biryani")))

    update = node({"user_query": "biryani near me"})

    assert update["cuisine"] == "biryani"
    assert [update[f] for f in ("location", "budget", "min_rating", "diet")] == [None] * 4


def test_remembered_preferences_returned_without_empty_fields():
    prefs = Preferences(
        cuisine="dosa", diet="vegetarian", remember=UserPreferences(diet="vegetarian")
    )
    update = make_extract_preferences(FakeModel(prefs))({"user_query": "I'm vegetarian, dosa?"})

    assert update["new_preferences"] == {"diet": "vegetarian"}
    assert update["diet"] == "vegetarian"


def test_first_request_sends_only_the_query():
    model = FakeModel(Preferences())
    make_extract_preferences(model)({"user_query": "dosa please"})

    system, human = model.calls[0]
    assert system.content == EXTRACT_PROMPT
    assert human.content == "dosa please"


def test_follow_up_includes_previous_search():
    model = FakeModel(Preferences())
    state = {
        "user_query": "something cheaper",
        "location": "Indiranagar",
        "cuisine": "South Indian",
        "budget": 800,
        "min_rating": None,
        "diet": None,
    }

    make_extract_preferences(model)(state)

    assert model.calls[0][1].content == (
        'Previous search: {"location": "Indiranagar", "cuisine": "South Indian", '
        '"budget": 800, "min_rating": null, "diet": null}\n\n'
        "New message: something cheaper"
    )


def test_new_chat_message_is_the_request():
    state = {"user_query": "old request",
             "messages": [HumanMessage("old request"), AIMessage("answer"), HumanMessage("dosa?")]}
    assert current_query(state) == "dosa?"


def test_user_query_used_when_no_new_chat_message():
    state = {"user_query": "biryani", "messages": [HumanMessage("dosa?"), AIMessage("answer")]}
    assert current_query(state) == "biryani"
    assert current_query({"user_query": "biryani"}) == "biryani"
