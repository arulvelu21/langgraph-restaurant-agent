"""Runtime settings, read from the environment (and .env if present)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use tools when they help."

# Default model per provider; override with AGENT_MODEL.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openai/gpt-oss-120b",
}

PROVIDER = os.getenv("AGENT_PROVIDER", "groq").lower()


@dataclass(frozen=True)
class Settings:
    provider: str = PROVIDER
    model: str = os.getenv("AGENT_MODEL") or DEFAULT_MODELS.get(PROVIDER, "")
    max_tokens: int = int(os.getenv("AGENT_MAX_TOKENS", "8000"))
    system_prompt: str = os.getenv("AGENT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    # Postgres URL for persistent memory; unset = in-memory (lost on exit).
    database_url: str | None = os.getenv("DATABASE_URL") or None


settings = Settings()
