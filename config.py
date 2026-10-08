from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator


class Settings(BaseModel):
    """Configuración validada del agente."""

    llm_provider: Literal["openai", "anthropic"] = "openai"

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    openai_model: str = "gpt-5-mini"
    anthropic_model: str = "claude-sonnet-4-6"

    checkpoint_db: Path = Path("./agent_checkpoints.sqlite")
    recursion_limit: int = Field(default=10, ge=4, le=30)

    @model_validator(mode="after")
    def validate_provider_key(self) -> "Settings":
        if self.llm_provider == "openai" and self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY es obligatoria cuando LLM_PROVIDER=openai."
            )

        if (
            self.llm_provider == "anthropic"
            and self.anthropic_api_key is None
        ):
            raise ValueError(
                "ANTHROPIC_API_KEY es obligatoria cuando "
                "LLM_PROVIDER=anthropic."
            )

        return self

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            anthropic_model=os.getenv(
                "ANTHROPIC_MODEL",
                "claude-sonnet-4-6",
            ),
            checkpoint_db=Path(
                os.getenv(
                    "CHECKPOINT_DB",
                    "./agent_checkpoints.sqlite",
                )
            ),
            recursion_limit=int(
                os.getenv("RECURSION_LIMIT", "10")
            ),
        )
