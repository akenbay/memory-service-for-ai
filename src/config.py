"""Centralized config from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://memory:memory@db:5432/memory"
    openai_api_key: str = ""
    extraction_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    memory_auth_token: str = ""


settings = Settings()