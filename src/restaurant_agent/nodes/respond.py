"""respond node: the last step on every path. Writes the outcome (top picks or
the reason there are none) as a Markdown chat message, for chat UIs and the CLI.
"""

from langchain_core.messages import AIMessage

from restaurant_agent.ranking import describe_place
from restaurant_agent.state import RestaurantState


def format_preferences(prefs: dict) -> str:
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in prefs.items()) or "none"


def format_answer(state: RestaurantState) -> str:
    parts = []
    if new := state.get("new_preferences"):
        parts.append(f"_Saved your preferences ({format_preferences(new)})._")

    recommendations = state.get("recommendations") or []
    if not recommendations:
        parts.append(state.get("message") or "No recommendations.")
    for i, r in enumerate(recommendations, start=1):
        lines = [f"**{i}. {r['name']}** · {describe_place(r)}"]
        if r.get("reason_source") == "llm":  # rules reasons repeat describe_place
            lines.append(r["reason"])
        if r.get("maps_url"):
            lines.append(f"[Open in Google Maps]({r['maps_url']})")
        parts.append("  \n".join(lines))
    return "\n\n".join(parts)


def respond(state: RestaurantState) -> dict:
    return {"messages": [AIMessage(format_answer(state))]}
