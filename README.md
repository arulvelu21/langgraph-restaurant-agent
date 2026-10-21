# Restaurant finder agent (LangGraph)

Ask for a restaurant in plain language and get real, filtered recommendations:

> Find me a good South Indian restaurant near me, rated above 4.2, under ₹800 per person.

The agent reads the request, searches Google Places, drops places that break your
limits, ranks what's left and recommends up to three, each with a reason and a
Google Maps link. It remembers standing preferences ("I'm vegetarian") per user, and
handles follow-ups ("something cheaper") within a conversation.

## How it works

The agent is a LangGraph `StateGraph`. Each step is a node that reads and updates a
shared state:

```
START → extract_preferences → apply_user_preferences
          ├─ "near me", location unknown → ask_for_location ──────────┐
          └─ search_restaurants                                        │
               ├─ nothing found → no_search_results ───────────────────┤
               └─ filter_restaurants                                   │
                    ├─ none left → no_matches ─────────────────────────┤
                    └─ rank_restaurants → llm_rank_restaurants         │
                         ├─ none relevant → no_relevant_matches ───────┤
                         └──────────────────────────────────────────── respond → END
```

| Step | What it does |
|---|---|
| `extract_preferences` | The LLM turns the request into structured fields (location, cuisine, budget, minimum rating, diet) plus any preferences to remember. Previous search fields are included, so follow-ups change the last search. |
| `apply_user_preferences` | Saves new standing preferences and fills empty fields from the user's saved ones. The request always wins. |
| `search_restaurants` | Google Places Text Search. Results are kept inside the searched area's map box (or a box around your default location for "near me"), so a rare cuisine returns nothing rather than restaurants in another city. |
| `filter_restaurants` | Plain Python: drops closed places, ratings below your minimum, and places whose per-person price is over budget. Unknown prices are kept. |
| `rank_restaurants` | Plain Python scoring: the rating is adjusted for review count (a 4.9 from 12 reviews doesn't beat a 4.6 from 3,000), with small penalties for distance and unknown price. Keeps a shortlist of 8. |
| `llm_rank_restaurants` | The LLM drops places that don't match (e.g. wrong cuisine), picks up to 3 and gives a one-sentence reason from the listed facts. If the LLM call fails, the score order is used instead. |
| `respond` | Writes the answer, or the reason there's nothing to show, as a Markdown chat message. |

The LLM handles language; ratings, distance and budget are deterministic Python, so they
behave the same on every run and are easy to test.

**State and memory.** A checkpointer keeps each conversation's state, which is how
follow-ups work. A store keeps each user's preferences across conversations. Both use
Postgres when `DATABASE_URL` is set, otherwise in-memory.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in your keys
```

In `.env`:

- `AGENT_PROVIDER`: `groq`, `openrouter` or `anthropic`, plus that provider's API key.
- `GOOGLE_PLACES_API_KEY`: a key with **Places API (New)** enabled.
- `DEFAULT_LOCATION` (optional): `lat,lng` used for "near me" searches.
- `DATABASE_URL` (optional): Postgres for persistent memory. Start it with `docker compose up -d`.

## Run

**Terminal chat**

```bash
uv run restaurant --user me --thread dinner
```

Reuse `--thread` to continue a conversation. `/prefs` shows saved preferences, `/forget` clears them.

**LangGraph server and Agent Chat UI**

```bash
uv run langgraph dev
```

This serves the `restaurant` graph from `langgraph.json`. Connect
[Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) or LangGraph Studio to it.

## Tests

```bash
uv run pytest tests/restaurant
```

The tests use a fake LLM and saved Google Places responses, so they need no API keys.
The Postgres persistence test runs when `DATABASE_URL` is set (start the database with `docker compose up -d`) and is skipped otherwise.

## Project layout

```
src/restaurant_agent/
  graph.py        builds the graph
  state.py        the shared state
  nodes/          one file per step
  places.py       Google Places client (the only code that calls Google)
  ranking.py      filtering and scoring, plain Python
  schemas.py      structured-output models
  prompts.py      LLM prompts
  llm.py          model for the chosen provider
  persistence.py  checkpointer and store (Postgres or in-memory)
  server.py       graph for `langgraph dev`
  cli.py          terminal chat
```
