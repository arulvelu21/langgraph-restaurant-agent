"""llm_rank_restaurants node: the LLM reads the user's request and the shortlist,
drops places that don't match (e.g. wrong cuisine), picks up to 3 and says why.
An empty result means nothing matched. If the LLM call fails, the shortlist's
own order is used, with reasons built from the data.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from restaurant_agent.prompts import RANK_PROMPT
from restaurant_agent.ranking import describe_place
from restaurant_agent.schemas import LLMRanking, Pick
from restaurant_agent.state import RestaurantState

log = logging.getLogger(__name__)

TOP_N = 3


REVIEW_SUMMARY_CHARS = 300  # keep the prompt short


def format_candidates(candidates: list[dict]) -> str:
    """One entry per candidate:
        1. Name [South Indian Restaurant] — 4.6★ (3,120 reviews) · ₹200–400 · Address
           Diners say: ...
    """
    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"{i}. {c.get('name')} [{c.get('category') or 'Restaurant'}] — "
            f"{describe_place(c)} · {c.get('address')}"
        )
        if summary := c.get("review_summary"):
            lines.append(f"   Diners say: {summary[:REVIEW_SUMMARY_CHARS]}")
    return "\n".join(lines)


def resolve_picks(picks: list[Pick], candidates: list[dict], top_n: int) -> list[dict]:
    """The LLM's picks as restaurants, ignoring unknown or repeated numbers."""
    chosen, seen = [], set()
    for pick in picks:
        index = pick.number - 1
        if 0 <= index < len(candidates) and index not in seen:
            seen.add(index)
            chosen.append({**candidates[index], "reason": pick.reason, "reason_source": "llm"})
    return chosen[:top_n]


def score_order(candidates: list[dict], top_n: int) -> list[dict]:
    """Fallback when the LLM fails: the shortlist as ranked, with data-based reasons."""
    return [
        {**c, "reason": describe_place(c), "reason_source": "rules"} for c in candidates[:top_n]
    ]


def make_llm_rank_restaurants(model: BaseChatModel, top_n: int = TOP_N):
    """Return the node, bound to `model` (pass a fake in tests)."""
    # nostream: keep this internal JSON out of chat UIs that stream LLM tokens
    ranker = model.with_structured_output(LLMRanking).with_config(tags=["nostream"])

    def llm_rank_restaurants(state: RestaurantState) -> dict:
        candidates = state.get("restaurants", [])
        if not candidates:
            return {"recommendations": []}

        request = f"Request: {state['user_query']}"
        if state.get("diet"):
            request += f"\nDietary need: {state['diet']}"
        try:
            ranking = ranker.invoke(
                [
                    SystemMessage(RANK_PROMPT.format(top_n=top_n)),
                    HumanMessage(f"{request}\n\nCandidates:\n{format_candidates(candidates)}"),
                ]
            )
            if ranking is None:  # replied in plain text instead of the structured output
                raise ValueError("LLM returned no structured ranking")
        except Exception:
            # Recommendations are still useful without the LLM's judgement.
            log.warning("LLM ranking failed; using score order", exc_info=True)
            return {"recommendations": score_order(candidates, top_n)}
        return {"recommendations": resolve_picks(ranking.picks, candidates, top_n)}

    return llm_rank_restaurants
