"""
config.py — Application settings loaded from environment variables / .env file.

Pydantic's BaseSettings reads from environment variables automatically.
If a .env file exists in the working directory, it's also loaded.
Missing required fields (no default) raise a clear error on startup — no silent misconfigs.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required — must be set in .env or environment
    deepseek_api_key: str
    internal_secret: str  # Shared secret between Spring Boot and this service

    # Optional — sensible defaults for local development
    spring_boot_url: str = "http://localhost:8080"

    # Conversation memory (Redis)
    redis_url: str = "redis://localhost:6379/0"
    conversation_ttl_seconds: int = 1800       # sliding TTL — refreshed on every turn
    conversation_max_messages: int = 10        # last N messages (5 user/assistant turns) kept per conversation

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
