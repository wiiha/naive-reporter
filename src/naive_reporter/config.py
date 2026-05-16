"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    docling_url: str = "http://localhost:5001"
    llm_api_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_model: str = "gemma4:31b-cloud"
    data_dir: str = "./data"


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
