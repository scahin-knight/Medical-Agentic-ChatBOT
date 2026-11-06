import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# SSL certificate fix for Python/Conda environments on Windows
os.environ["SSL_CERT_FILE"] = certifi.where()

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    # Local Ollama models (embeddings are set in app/knowledge.py)
    ollama_chat_model: str = "llama3.1:8b"
    ollama_vision_model: str = "moondream"

    pinecone_api_key: str
    tavily_api_key: str

    api_token: str
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()