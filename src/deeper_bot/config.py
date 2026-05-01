import os
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    bot_token: str

    llm_base_url: str
    llm_model: str
    llm_api_key_env: str
    llm_use_reasoning: bool = True
    llm_reasoning_effort: str = "high"
    llm_utility_model: str | None = None

    allowed_users: list[int] = []
    database_path: str = "data/sessions.db"

    @field_validator("bot_token", mode="after")
    @classmethod
    def validate_bot_token(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("BOT_TOKEN must be set — get one from @BotFather on Telegram")
        return v

    @field_validator("allowed_users", mode="before")
    @classmethod
    def parse_allowed_users(cls, v: object) -> list[int]:
        if isinstance(v, str):
            if not v.strip():
                return []
            return [int(uid.strip()) for uid in v.split(",") if uid.strip()]
        return v  # type: ignore[return-value]

    @property
    def llm_api_key(self) -> str:
        try:
            return os.environ[self.llm_api_key_env]
        except KeyError:
            raise RuntimeError(
                f"Environment variable '{self.llm_api_key_env}' is not set. "
                f"Set it in your .env file or export it in your shell."
            ) from None

    @property
    def resolved_utility_model(self) -> str:
        return self.llm_utility_model or self.llm_model


@lru_cache
def get_settings() -> Settings:
    # BaseSettings populates fields from env vars / .env automatically.
    return Settings()  # type: ignore[call-arg]
