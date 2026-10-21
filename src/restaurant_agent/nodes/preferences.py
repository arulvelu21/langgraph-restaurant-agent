"""apply_user_preferences node: saves any new standing preferences for the user,
then fills search fields the request left empty from their saved preferences.

Preferences live in the LangGraph store under ("users", user_id), so they are
shared by all of a user's threads. Without a store, nothing is saved or applied.
"""

from langgraph.runtime import Runtime

from restaurant_agent.state import Context, RestaurantState

KEY = "preferences"


def namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id)


def apply_user_preferences(state: RestaurantState, runtime: Runtime[Context]) -> dict:
    store = runtime.store
    if store is None:
        return {"user_preferences": {}}

    # No context passed to invoke() (e.g. from Studio) -> the default user
    ns = namespace((runtime.context or Context()).user_id)
    item = store.get(ns, KEY)
    saved = item.value if item else {}
    if new := state.get("new_preferences"):
        saved = {**saved, **new}
        store.put(ns, KEY, saved)

    update: dict = {"user_preferences": saved}
    # The request wins; saved preferences only fill what it left empty.
    if state.get("location") is None and saved.get("home_location"):
        update["location"] = saved["home_location"]
    for field in ("diet", "budget", "min_rating"):
        if state.get(field) is None and saved.get(field) is not None:
            update[field] = saved[field]
    return update
