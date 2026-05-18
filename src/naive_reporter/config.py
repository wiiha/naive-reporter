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
    llm_api_key: str = "local-key"
    llm_model: str = "gemma4:31b-cloud"
    data_dir: str = "./data"
    llm_reporting_model: str = "deepseek-v4-flash:cloud"
    llm_reporting_api_url: str = ""
    llm_reporting_api_key: str = ""
    language_directive: str = ""
    number_of_synthetic_user_queries: int = 5

    def get_reporting_client_config(self) -> dict[str, str]:
        """Resolved reporting LLM config with fallback to main LLM."""
        return {
            "base_url": self.llm_reporting_api_url or self.llm_api_url,
            "api_key": self.llm_reporting_api_key or self.llm_api_key,
            "model": self.llm_reporting_model,
        }


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
