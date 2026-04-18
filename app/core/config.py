from functools import lru_cache

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = Field(default="Decision_Insight_POC")
    ENVIRONMENT: str = Field(default="dev")

    # CORS — stored as a plain comma-separated string.
    # Kept as str intentionally: pydantic-settings v2 runs json.loads() on
    # list fields before validators fire, which breaks non-JSON env values.
    # Parsed into a list in main.py where it is consumed.
    # e.g. CORS_ALLOWED_ORIGINS=http://localhost:5173,https://yourdomain.com
    CORS_ALLOWED_ORIGINS: str = Field(default="http://localhost:5173")

    # -------------------------------------------------------------------------
    # Azure Blob Storage — required, no defaults (fails fast if missing)
    # -------------------------------------------------------------------------
    AZURE_STORAGE_CONNECTION_STRING: str = Field(...)
    AZURE_STORAGE_CONTAINER_NAME: str = Field(...)

    # -------------------------------------------------------------------------
    # Azure OpenAI — required for embedding generation
    # -------------------------------------------------------------------------
    AZURE_OPENAI_ENDPOINT: str = Field(...)
    AZURE_OPENAI_API_KEY: str = Field(...)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(...)
    AZURE_OPENAI_API_VERSION: str = Field(...)  

    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(...)
    AZURE_OPENAI_CHAT_API_VERSION: str = Field(...)

    # -------------------------------------------------------------------------
    # Azure AI Search — required for document indexing
    # -------------------------------------------------------------------------
    AZURE_SEARCH_ENDPOINT: str = Field(...)
    AZURE_SEARCH_API_KEY: str = Field(...)
    AZURE_SEARCH_INDEX_NAME: str = Field(...)

    # -------------------------------------------------------------------------
    # Azure AI Foundry Agent — required for chat (replaces direct LLM calls)
    # Auth: DefaultAzureCredential — run "az login" locally; Managed Identity in prod
    # -------------------------------------------------------------------------
    AZURE_AI_PROJECT_ENDPOINT: str = Field(...)
    AZURE_AI_AGENT_NAME: str = Field(...)

    model_config = {
        "env_file": ".env",       # local dev only; override via env vars in production
        "case_sensitive": True,
        "extra": "ignore",        # silently ignore any unrecognised env vars
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Load and cache application settings.
    Fails fast on startup if any required environment variable is missing.
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise exc
