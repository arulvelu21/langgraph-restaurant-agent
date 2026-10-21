"""Runtime settings, read from the environment (and .env if present).

Uses the same AGENT_* variables as the agent package, so both share one .env.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Default model per provider; override with AGENT_MODEL.
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openai/gpt-oss-120b",
}

PROVIDER = os.getenv("AGENT_PROVIDER", "groq").lower()


def parse_lat_lng(value: str | None) -> tuple[float, float] | None:
    """'12.97,77.59' -> (12.97, 77.59); None/empty -> None."""
    if not value:
        return None
    try:
        lat, lng = (float(part) for part in value.split(","))
    except ValueError:
        raise ValueError(f"DEFAULT_LOCATION must be 'lat,lng', got {value!r}") from None
    return lat, lng


@dataclass(frozen=True)
class Settings:
    provider: str = PROVIDER
    model: str = os.getenv("AGENT_MODEL") or DEFAULT_MODELS.get(PROVIDER, "")
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "8000"))
    google_places_api_key: str | None = os.getenv("GOOGLE_PLACES_API_KEY") or None
    # Where "near me" searches are centred; unset = "near me" requests fail.
    default_location: tuple[float, float] | None = parse_lat_lng(os.getenv("DEFAULT_LOCATION"))
    search_radius_m: int = int(os.getenv("SEARCH_RADIUS_M", "5000"))
    # Postgres for checkpoints and user preferences; unset = in-memory (lost on exit).
    database_url: str | None = os.getenv("DATABASE_URL") or None


settings = Settings()
