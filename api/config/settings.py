from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    qdrant_url: str = Field(validation_alias="qdrant_host")
    qdrant_api_key: str
    collection_name: str = "financial"

    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "Qdrant/bm25"
    colbert_model: str = "colbert-ir/colbertv2.0"

    groq_api_key: str = Field(validation_alias="openai_api_key")
    # groq_model_name: str = "llama-3.1-8b-instant"
    groq_model_name: str = "gpt-4.1-mini"


settings = Settings()
