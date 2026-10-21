"""Tests for the messages shown when there is nothing to recommend."""

from restaurant_agent.nodes.no_results import (
    no_matches,
    no_relevant_matches,
    no_search_results,
)


def test_no_search_results_near_me_without_cuisine():
    update = no_search_results({"location": None, "cuisine": None})

    assert update == {
        "recommendations": [],
        "message": "Google didn't find any restaurants near you. "
        "Try a broader cuisine or a different area.",
    }


def test_no_matches_only_rating():
    update = no_matches({"min_rating": 4.5, "budget": None})

    assert update["message"] == (
        "Google found restaurants, but none are open with a rating of 4.5 or more. "
        "Try lowering the minimum rating."
    )


def test_no_matches_without_limits_means_all_closed():
    update = no_matches({"min_rating": None, "budget": None})

    assert update["message"] == (
        "Google found restaurants, but none of them are currently open for business."
    )


def test_no_relevant_matches():
    update = no_relevant_matches({"location": "Hampi", "cuisine": "Ethiopian"})

    assert update == {
        "recommendations": [],
        "message": "Couldn't find Ethiopian restaurants in Hampi. Google only returned other "
        "kinds of restaurants, so none are shown. Try a broader cuisine or a different area.",
    }


def test_no_relevant_matches_without_cuisine():
    update = no_relevant_matches({"location": None, "cuisine": None})

    assert update["message"].startswith(
        "Couldn't find restaurants that match your request near you."
    )
