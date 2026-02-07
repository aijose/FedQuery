"""LLM provider configuration using LangChain abstractions."""

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import get_settings


def get_llm() -> BaseChatModel:
    """Create and return the configured LLM instance.

    Uses LangChain's BaseChatModel abstraction for LLM-agnostic access.
    Default: Anthropic Claude via langchain-anthropic.
    """
    settings = get_settings()
    provider = settings.fedquery_llm_provider.lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.fedquery_llm_model,
            temperature=0,
            api_key=settings.anthropic_api_key,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.fedquery_llm_model,
            temperature=0,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "Supported: 'anthropic', 'google'"
        )
