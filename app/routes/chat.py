"""Endpoints de conversación y estado del servicio."""

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.llm.ollama_client import LLMError, OllamaChat
from app.models import ChatRequest, HealthResponse
from app.prompts import construir_mensajes
from app.rag.embeddings import EmbeddingError
from app.rag.retriever import Retriever
from app.session import SessionStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


def _sse(evento: str, datos: dict) -> str:
    """Serializa un evento en formato Server-Sent Events."""
    return f"event: {evento}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(peticion: ChatRequest, request: Request) -> StreamingResponse:
    """Responde una consulta en streaming (SSE).

    Eventos emitidos, en orden:
    - `session`: id de la conversación, para que el cliente lo reenvíe.
    - `sources`: documentos recuperados que se usaron como contexto.
    - `token`: fragmentos de texto, a medida que el modelo los genera.
    - `done`: fin de la respuesta.
    - `error`: algo falló; el mensaje es apto para mostrarle al usuario.
    """
    estado = request.app.state
    retriever: Retriever | None = estado.retriever
    llm: OllamaChat = estado.llm
    sesiones: SessionStore = estado.sessions

    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="El índice de datos no está disponible. Revisá los logs del servidor.",
        )

    sesion = sesiones.obtener_o_crear(peticion.session_id)
    pregunta = peticion.content.strip()

    async def generar() -> AsyncIterator[str]:
        yield _sse("session", {"session_id": sesion.id})

        try:
            resultados = retriever.search(pregunta, top_k=estado.settings.top_k)
        except EmbeddingError as exc:
            logger.error("Falló el retrieval: %s", exc)
            yield _sse("error", {"message": str(exc)})
            return

        yield _sse(
            "sources",
            {
                "sources": [
                    {
                        "source": r.document.source,
                        "text": r.document.text,
                        "score": round(r.score, 4),
                    }
                    for r in resultados
                ]
            },
        )

        mensajes = construir_mensajes(pregunta, sesiones.historial(sesion.id), resultados)

        partes: list[str] = []
        try:
            async for fragmento in llm.stream(mensajes):
                partes.append(fragmento)
                yield _sse("token", {"text": fragmento})
        except LLMError as exc:
            logger.error("Falló la generación: %s", exc)
            yield _sse("error", {"message": str(exc)})
            return

        respuesta = "".join(partes)
        # El historial guarda la pregunta cruda, sin el bloque de contexto: el
        # contexto se recupera de nuevo en cada turno y repetirlo desperdicia
        # ventana de contexto.
        sesiones.agregar(sesion.id, "user", pregunta)
        sesiones.agregar(sesion.id, "assistant", respuesta)

        yield _sse("done", {"session_id": sesion.id})

    return StreamingResponse(
        generar(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Estado de las dependencias: Ollama, modelos e índice de documentos."""
    estado = request.app.state
    llm: OllamaChat = estado.llm
    settings = estado.settings

    documentos = len(estado.retriever.documentos) if estado.retriever else 0

    try:
        modelos = await llm.modelos_disponibles()
    except LLMError as exc:
        return HealthResponse(
            status="degraded",
            ollama=False,
            chat_model=settings.chat_model,
            chat_model_disponible=False,
            embedding_model=settings.embedding_model,
            documentos_indexados=documentos,
            detalle=str(exc),
        )

    # Ollama reporta "llama3.2:3b"; aceptamos también que el usuario configure "llama3.2".
    chat_ok = any(
        m == settings.chat_model or m.startswith(f"{settings.chat_model}:") for m in modelos
    )

    detalle = None
    if not chat_ok:
        detalle = (
            f"El modelo '{settings.chat_model}' no está descargado. "
            f"Corré `ollama pull {settings.chat_model}`."
        )
    elif documentos == 0:
        detalle = "No hay documentos indexados."

    return HealthResponse(
        status="ok" if chat_ok and documentos > 0 else "degraded",
        ollama=True,
        chat_model=settings.chat_model,
        chat_model_disponible=chat_ok,
        embedding_model=settings.embedding_model,
        documentos_indexados=documentos,
        detalle=detalle,
    )
