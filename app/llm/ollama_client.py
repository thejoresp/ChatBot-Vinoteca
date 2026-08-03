"""Cliente asíncrono de Ollama para generación en streaming."""

import logging
from collections.abc import AsyncIterator

import ollama

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """El modelo de lenguaje no está disponible o falló durante la generación."""


class OllamaChat:
    """Envuelve `ollama.AsyncClient` para exponer sólo lo que la app necesita."""

    def __init__(self, model: str, host: str, temperature: float = 0.2) -> None:
        self.model = model
        self.host = host
        self.temperature = temperature
        self._client = ollama.AsyncClient(host=host)

    async def stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Genera la respuesta token a token.

        Se genera con temperatura baja a propósito: la tarea es reproducir datos
        del catálogo, no redactar con creatividad. Con la temperatura por defecto
        (0.8), el mismo prompt daba respuestas de calidad muy distinta entre
        corridas —a veces omitía productos de la lista, a veces negaba una
        sucursal que sí estaba en el contexto—, lo que además hace imposible
        evaluar un cambio de prompt.

        Yields:
            Fragmentos de texto en el orden en que los produce el modelo.

        Raises:
            LLMError: si Ollama no responde o el modelo no está descargado.
        """
        try:
            respuesta = await self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": self.temperature},
            )
            async for chunk in respuesta:
                contenido = chunk.get("message", {}).get("content", "")
                if contenido:
                    yield contenido
        except ollama.ResponseError as exc:
            raise LLMError(
                f"Ollama rechazó la consulta al modelo '{self.model}': {exc}. "
                f"Si el modelo no está descargado, corré `ollama pull {self.model}`."
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"No se pudo contactar a Ollama en {self.host}: {exc}. "
                "Verificá que el servidor esté corriendo (`ollama serve`)."
            ) from exc

    async def modelos_disponibles(self) -> list[str]:
        """Lista los modelos descargados en el servidor.

        Raises:
            LLMError: si el servidor no responde.
        """
        try:
            respuesta = await self._client.list()
        except Exception as exc:
            raise LLMError(f"No se pudo contactar a Ollama en {self.host}: {exc}") from exc
        return [m.get("model", "") for m in respuesta.get("models", [])]
