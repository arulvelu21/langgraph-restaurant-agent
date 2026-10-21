"""Tests for the respond node: the final chat answer."""

from restaurant_agent.nodes.respond import format_answer, respond

REC = {"name": "Test Dosa House", "rating": 4.6, "review_count": 3120, "price_min": 200,
       "price_max": 400, "price_currency": "INR", "reason": "Diners praise the ghee dosa.",
       "reason_source": "llm", "maps_url": "https://maps.google.com/?cid=1"}


def test_recommendations_as_markdown():
    answer = format_answer({"recommendations": [REC, {**REC, "name": "Second", "maps_url": None}]})

    assert answer == (
        "**1. Test Dosa House** · 4.6★ (3,120 reviews) · ₹200–400  \n"
        "Diners praise the ghee dosa.  \n"
        "[Open in Google Maps](https://maps.google.com/?cid=1)\n\n"
        "**2. Second** · 4.6★ (3,120 reviews) · ₹200–400  \n"
        "Diners praise the ghee dosa."
    )


def test_rules_reason_not_repeated():
    rec = {**REC, "reason": "4.6★ (3,120 reviews) · ₹200–400", "reason_source": "rules"}
    assert "Diners" not in format_answer({"recommendations": [rec]})
    assert format_answer({"recommendations": [rec]}).count("3,120") == 1


def test_message_and_saved_preferences():
    answer = format_answer({"recommendations": [], "message": "Where should I look?",
                            "new_preferences": {"diet": "vegetarian"}})

    assert answer == "_Saved your preferences (diet: vegetarian)._\n\nWhere should I look?"


def test_respond_adds_an_ai_message():
    (message,) = respond({"recommendations": [], "message": "Nothing found."})["messages"]
    assert message.type == "ai" and message.content == "Nothing found."
