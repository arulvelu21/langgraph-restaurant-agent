"""Builds the chat model for AGENT_PROVIDER (imports only that provider's package)."""

from langchain_core.language_models import BaseChatModel

from restaurant_agent.config import settings


def get_model() -> BaseChatModel:
    # temperature 0: extraction should give the same answer for the same request
    kwargs = {"model": settings.model, "max_tokens": settings.max_tokens, "temperature": 0}
    if settings.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(**kwargs)
    if settings.provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(**kwargs)
    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kwargs)
    raise ValueError(
        f"Unknown AGENT_PROVIDER {settings.provider!r}; use groq, openrouter or anthropic."
    )
