"""Punto de entrada de la aplicación FastAPI.

Se levanta con:
    uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, get_settings
from app.llm.ollama_client import OllamaChat
from app.rag.embeddings import EmbeddingError, OllamaEmbeddings
from app.rag.loader import cargar_documentos
from app.rag.retriever import CosineRetriever
from app.routes.chat import router as chat_router
from app.session import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carga los datos y construye el índice al arrancar.

    Los errores de datos son fatales (sin datos no hay asistente), pero los de
    Ollama no: el servidor levanta igual y `/api/health` explica qué falta, para
    que el usuario pueda arrancar Ollama sin reiniciar la app.
    """
    settings = get_settings()
    app.state.settings = settings
    app.state.sessions = SessionStore(
        max_turns=settings.max_history_turns, ttl_seconds=settings.session_ttl_seconds
    )
    app.state.llm = OllamaChat(
        model=settings.chat_model,
        host=settings.ollama_host,
        temperature=settings.temperature,
    )
    app.state.retriever = None

    documentos = cargar_documentos(settings.data_dir)
    logger.info("Documentos cargados: %s", len(documentos))

    provider = OllamaEmbeddings(model=settings.embedding_model, host=settings.ollama_host)
    try:
        app.state.retriever = CosineRetriever.desde_documentos(
            documentos,
            provider,
            cache_dir=settings.cache_dir,
            model_name=settings.embedding_model,
            threshold=settings.similarity_threshold,
        )
        logger.info("Índice listo. La API está disponible en /api/chat")
    except EmbeddingError as exc:
        logger.error("No se pudo construir el índice: %s", exc)
        logger.error("El servidor arranca igual; revisá GET /api/health")

    yield


app = FastAPI(
    title="Enotek Vinos - Asistente",
    description=(
        "Asistente conversacional de vinoteca. Responde sobre precios, sucursales y "
        "horarios usando RAG sobre datos propios y un LLM local vía Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()

# El frontend se sirve desde este mismo origen, así que CORS sólo hace falta si
# alguien consume la API desde otra app. Lista explícita, no comodín.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(chat_router)

# Montado al final para que no tape las rutas de /api.
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


def main() -> None:
    """Arranca el servidor de desarrollo."""
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
