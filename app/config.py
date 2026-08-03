"""Configuración de la aplicación, leída de variables de entorno o `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del repositorio. Todos los paths se derivan de acá para que el proyecto
# funcione en cualquier máquina sin editar código.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Parámetros configurables. Los defaults son válidos para correr en local."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Ollama ---
    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    # Baja a propósito: la tarea es reproducir datos del catálogo, no redactar
    # con creatividad. Con el default de Ollama (0.8) las respuestas varían
    # demasiado entre corridas para el mismo prompt.
    temperature: float = 0.2

    # --- RAG ---
    data_dir: Path = BASE_DIR / "data"
    cache_dir: Path = BASE_DIR / ".cache"
    top_k: int = 5
    # Piso de similitud coseno para incluir un documento. Calibrado sobre
    # nomic-embed-text, donde las consultas del dominio puntúan 0.62-0.79: sirve
    # para recortar la cola floja de cada búsqueda, no para filtrar preguntas
    # fuera de tema (ver la nota en `rag/retriever.py`).
    similarity_threshold: float = 0.55

    # --- Conversación ---
    # Turnos (par usuario+asistente) que se recuerdan por sesión.
    max_history_turns: int = 10
    session_ttl_seconds: int = 3600

    # --- API ---
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración cacheada (se lee una sola vez por proceso)."""
    return Settings()
