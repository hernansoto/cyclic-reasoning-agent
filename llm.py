from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    """Crea el LLM seleccionado; la lógica del grafo no cambia por proveedor."""
    if settings.llm_provider == "openai":
        assert settings.openai_api_key is not None

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    assert settings.anthropic_api_key is not None

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
    )
