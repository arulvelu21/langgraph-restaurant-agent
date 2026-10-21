"""Tests for the llm_rank_restaurants node, using a fake model (no API calls)."""

from langchain_core.runnables import RunnableLambda

from restaurant_agent.nodes.llm_rank import make_llm_rank_restaurants
from restaurant_agent.schemas import LLMRanking, Pick


class FakeModel:
    """Returns a fixed LLMRanking (or raises) and records the messages it was sent."""

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is LLMRanking

        def respond(messages):
            self.calls.append(messages)
            if self.error:
                raise self.error
            return self.result

        return RunnableLambda(respond)


CANDIDATES = [  # no "category": shown as the generic [Restaurant]
    {"id": f"p{i}", "name": f"Place {i}", "address": f"{i} Road", "rating": 4.5,
     "review_count": 100 * i, "price_min": 200, "price_max": 400, "price_currency": "INR"}
    for i in range(1, 6)
]
STATE = {"user_query": "South Indian near me under ₹800", "restaurants": CANDIDATES}


def ranking(*numbers):
    return LLMRanking(picks=[Pick(number=n, reason=f"reason {n}") for n in numbers])


def ids(recs):
    return [r["id"] for r in recs]


def test_uses_llm_order_and_reasons():
    node = make_llm_rank_restaurants(FakeModel(ranking(3, 1, 5)))

    recs = node(STATE)["recommendations"]

    assert ids(recs) == ["p3", "p1", "p5"]
    assert [r["reason"] for r in recs] == ["reason 3", "reason 1", "reason 5"]
    assert {r["reason_source"] for r in recs} == {"llm"}


def test_bad_and_repeated_numbers_ignored_without_filling():
    node = make_llm_rank_restaurants(FakeModel(ranking(4, 99, 4, 0)))

    assert ids(node(STATE)["recommendations"]) == ["p4"]


def test_no_picks_means_nothing_matched():
    node = make_llm_rank_restaurants(FakeModel(ranking()))

    assert node(STATE) == {"recommendations": []}


def test_more_picks_than_top_n_are_cut():
    node = make_llm_rank_restaurants(FakeModel(ranking(5, 4, 3, 2)))

    assert ids(node(STATE)["recommendations"]) == ["p5", "p4", "p3"]


def test_llm_error_falls_back_to_shortlist_order():
    node = make_llm_rank_restaurants(FakeModel(error=RuntimeError("rate limited")))

    recs = node(STATE)["recommendations"]

    assert ids(recs) == ["p1", "p2", "p3"]
    assert {r["reason_source"] for r in recs} == {"rules"}
    assert recs[0]["reason"] == "4.5★ (100 reviews) · ₹200–400"


def test_plain_text_reply_falls_back_to_shortlist_order():
    # with_structured_output gives None when the model answers in text, not the schema
    node = make_llm_rank_restaurants(FakeModel(result=None))

    assert ids(node(STATE)["recommendations"]) == ["p1", "p2", "p3"]


def test_fewer_candidates_than_top_n():
    node = make_llm_rank_restaurants(FakeModel(ranking(2, 1)))

    recs = node({**STATE, "restaurants": CANDIDATES[:2]})["recommendations"]

    assert ids(recs) == ["p2", "p1"]


def test_no_candidates_skips_llm():
    model = FakeModel(ranking(1))

    assert make_llm_rank_restaurants(model)({**STATE, "restaurants": []}) == {"recommendations": []}
    assert model.calls == []


def test_sends_request_and_numbered_candidates():
    model = FakeModel(ranking(1))
    make_llm_rank_restaurants(model)(STATE)

    system, human = model.calls[0]
    assert "pick up to 3" in system.content
    assert human.content.startswith("Request: South Indian near me under ₹800")
    assert "1. Place 1 [Restaurant] — 4.5★ (100 reviews) · ₹200–400 · 1 Road" in human.content
    assert "5. Place 5" in human.content


def test_candidates_include_review_summary():
    model = FakeModel(ranking(1))
    with_summary = {**CANDIDATES[0], "category": "South Indian Restaurant",
                    "review_summary": "Diners praise the ghee dosa."}

    make_llm_rank_restaurants(model)({**STATE, "restaurants": [with_summary, CANDIDATES[1]]})

    human = model.calls[0][1].content
    assert ("1. Place 1 [South Indian Restaurant] — 4.5★ (100 reviews) · ₹200–400 · 1 Road\n"
            "   Diners say: Diners praise the ghee dosa.\n"
            "2. Place 2 [Restaurant]") in human
