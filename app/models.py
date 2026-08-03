"""Esquemas de entrada y salida de la API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Consulta del usuario."""

    content: str = Field(..., min_length=1, max_length=2000, description="Pregunta del usuario")
    session_id: str | None = Field(
        None, description="Identificador de la conversación. Si se omite, el servidor crea una."
    )


class FuenteCitada(BaseModel):
    """Documento que se usó como contexto para responder."""

    source: str = Field(..., description="Origen del dato: 'precios' o 'ubicaciones'")
    text: str = Field(..., description="Texto del documento recuperado")
    score: float = Field(..., description="Similitud coseno con la consulta")


class HealthResponse(BaseModel):
    """Estado de las dependencias de la aplicación."""

    status: str = Field(..., description="'ok' si todo funciona, 'degraded' si algo falta")
    ollama: bool = Field(..., description="El servidor de Ollama responde")
    chat_model: str
    chat_model_disponible: bool
    embedding_model: str
    documentos_indexados: int
    detalle: str | None = None
