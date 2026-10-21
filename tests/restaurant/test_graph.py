"""Tests for the whole graph with a fake LLM and fake Places client, including
the two "nothing to recommend" branches.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from restaurant_agent.graph import build_graph
from restaurant_agent.schemas import LLMRanking, Pick, Preferences, UserPreferences
from restaurant_agent.state import Context


class FakeModel:
    """Answers each structured-output schema with a preset value; records LLM calls.
    `preferences` can be a list to give a different answer on each request.
    """

    def __init__(self, preferences, ranking: LLMRanking | None = None):
        self.preferences = preferences if isinstance(preferences, list) else None
        self.answers = {Preferences: preferences, LLMRanking: ranking or LLMRanking(picks=[])}
        self.calls = []
        self.messages = []

    def with_structured_output(self, schema):
        def respond(messages):
            self.calls.append(schema)
            self.messages.append(messages)
            if schema is Preferences and self.preferences is not None:
                return self.preferences.pop(0)
            return self.answers[schema]

        return RunnableLambda(respond)


class FakePlaces:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search_text(self, query, **kwargs):
        self.queries.append(query)
        return self.results

    def find_area(self, name, near=None):
        return ((12.9, 77.6), (13.0, 77.7))


def place(id, rating=4.5, price_min=200):
    return {"id": id, "name": f"Place {id}", "address": "Indiranagar", "rating": rating,
            "review_count": 1000, "price_min": price_min, "price_max": price_min * 2,
            "price_currency": "INR", "price_level": None, "business_status": "OPERATIONAL",
            "lat": None, "lng": None}


PREFS = Preferences(location="Indiranagar", cuisine="South Indian", budget=800, min_rating=4.2)


def run(model, places):
    return build_graph(model=model, places_client=FakePlaces(places)).invoke(
        {"user_query": "South Indian in Indiranagar, rating > 4.2, under ₹800"}
    )


def test_happy_path_returns_top_3():
    picks = [Pick(number=n, reason=f"reason {n}") for n in (2, 1, 3)]
    picks[0].reason = "best dosa"
    model = FakeModel(PREFS, LLMRanking(picks=picks))

    result = run(model, [place("a"), place("b", rating=4.8), place("c"), place("d")])

    assert len(result["recommendations"]) == 3
    assert result["recommendations"][0]["reason"] == "best dosa"
    assert result.get("message") is None
    assert model.calls == [Preferences, LLMRanking]


def test_google_returns_nothing():
    model = FakeModel(PREFS)

    result = run(model, [])

    assert result["recommendations"] == []
    assert result["message"] == (
        "Google didn't find any South Indian restaurants in Indiranagar. "
        "Try a broader cuisine or a different area."
    )
    assert model.calls == [Preferences]  # never reached the LLM ranking


def test_everything_filtered_out():
    model = FakeModel(PREFS)

    result = run(model, [place("low", rating=3.9), place("pricey", price_min=1500)])

    assert result["recommendations"] == []
    assert result["message"] == (
        "Google found restaurants, but none are open with a rating of 4.2 or more "
        "and a price under ₹800 per person. Try lowering the minimum rating or raising the budget."
    )
    assert model.calls == [Preferences]


def test_llm_finds_none_relevant():
    model = FakeModel(PREFS, LLMRanking(picks=[]))

    result = run(model, [place("a"), place("b")])

    assert result["recommendations"] == []
    assert result["message"].startswith("Couldn't find South Indian restaurants in Indiranagar.")
    assert model.calls == [Preferences, LLMRanking]


def test_preferences_saved_in_one_thread_apply_in_another():
    model = FakeModel([
        Preferences(location="Gunjur", cuisine="dosa", diet="vegetarian",
                    remember=UserPreferences(diet="vegetarian")),
        Preferences(location="Gunjur", cuisine="biryani"),
        Preferences(location="Gunjur", cuisine="biryani"),
    ], LLMRanking(picks=[Pick(number=1, reason="r")]))
    places = FakePlaces([place("a")])
    graph = build_graph(model=model, places_client=places, checkpointer=InMemorySaver(),
                        store=InMemoryStore())

    def ask(query, thread, user="arul"):
        return graph.invoke({"user_query": query}, {"configurable": {"thread_id": thread}},
                            context=Context(user_id=user))

    first = ask("I'm vegetarian, dosa in Gunjur", "t1")
    second = ask("biryani in Gunjur", "t2")
    other_user = ask("biryani in Gunjur", "t3", user="priya")

    assert first["new_preferences"] == {"diet": "vegetarian"}
    assert second["diet"] == "vegetarian"
    assert second["user_preferences"] == {"diet": "vegetarian"}
    assert other_user["diet"] is None
    assert places.queries == [
        "vegetarian dosa restaurant in Gunjur",
        "vegetarian biryani restaurant in Gunjur",
        "biryani restaurant in Gunjur",
    ]


def test_checkpoint_keeps_thread_state_for_follow_ups():
    model = FakeModel([PREFS, PREFS.model_copy(update={"budget": 500})],
                      LLMRanking(picks=[Pick(number=1, reason="r")]))
    graph = build_graph(model=model, places_client=FakePlaces([place("a")]),
                        checkpointer=InMemorySaver(), store=InMemoryStore())
    config = {"configurable": {"thread_id": "t1"}}

    graph.invoke({"user_query": "South Indian in Indiranagar under ₹800"}, config)
    second = graph.invoke({"user_query": "something cheaper"}, config)

    # The second extract call saw the first search
    follow_up = model.messages[2][1].content  # [extract, rank, extract, ...]
    assert follow_up.startswith('Previous search: {"location": "Indiranagar"')
    assert follow_up.endswith("New message: something cheaper")
    assert second["budget"] == 500


def test_message_from_a_failed_request_is_cleared():
    model = FakeModel([PREFS, PREFS], LLMRanking(picks=[Pick(number=1, reason="r")]))
    places = FakePlaces([])
    graph = build_graph(model=model, places_client=places, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    assert graph.invoke({"user_query": "q1"}, config)["message"]
    places.results = [place("a")]
    assert graph.invoke({"user_query": "q2"}, config)["message"] is None


def test_near_me_without_any_location_asks_where():
    model = FakeModel(Preferences(cuisine="dosa"))
    places = FakePlaces([place("a")])
    graph = build_graph(model=model, places_client=places, default_location=None)

    result = graph.invoke({"user_query": "dosa near me"})

    assert result["recommendations"] == []
    assert result["message"].startswith("Where should I look?")
    assert places.queries == []  # never searched


def test_near_me_uses_saved_home_without_default_location():
    model = FakeModel([
        Preferences(remember=UserPreferences(home_location="Gunjur, Bangalore")),
        Preferences(cuisine="dosa"),
    ], LLMRanking(picks=[Pick(number=1, reason="r")]))
    places = FakePlaces([place("a")])
    graph = build_graph(model=model, places_client=places, checkpointer=InMemorySaver(),
                        store=InMemoryStore(), default_location=None)
    config = {"configurable": {"thread_id": "t1"}}

    graph.invoke({"user_query": "I live in Gunjur, Bangalore"}, config)
    result = graph.invoke({"user_query": "dosa near me"}, config)

    assert places.queries[-1] == "dosa restaurant in Gunjur, Bangalore"
    assert result["recommendations"][0]["id"] == "a"


def test_chat_messages_in_and_answer_out():
    """How Agent Chat UI and the CLI call the graph."""
    model = FakeModel([PREFS, PREFS], LLMRanking(picks=[Pick(number=1, reason="best dosa")]))
    places = FakePlaces([place("a")])
    graph = build_graph(model=model, places_client=places, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    first = graph.invoke({"messages": [HumanMessage("dosa in Indiranagar")]}, config)
    places.results = []
    second = graph.invoke({"messages": [HumanMessage("something cheaper")]}, config)

    assert first["user_query"] == "dosa in Indiranagar"
    assert "**1. Place a**" in first["messages"][-1].content
    assert "best dosa" in first["messages"][-1].content
    assert second["user_query"] == "something cheaper"
    assert [m.type for m in second["messages"]] == ["human", "ai", "human", "ai"]
    assert second["messages"][-1].content.startswith("Google didn't find any")
