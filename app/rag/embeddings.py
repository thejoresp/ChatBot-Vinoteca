"""Generación de embeddings vía Ollama, con cache en disco.

Los embeddings se calculan localmente con `nomic-embed-text`: no hay llamadas a
servicios externos ni API keys. Como el catálogo cambia poco, los vectores se
cachean en disco y sólo se recalculan si el contenido de los documentos cambió.
"""

import hashlib
import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import ollama

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """No se pudieron generar los embeddings."""


class EmbeddingProvider(Protocol):
    """Contrato mínimo para generar embeddings.

    Los documentos y las consultas se embeben por separado porque los modelos
    asimétricos (como nomic-embed-text) esperan un prefijo distinto para cada
    rol. Existe además para que los tests puedan inyectar un proveedor
    determinista y para poder cambiar de modelo sin tocar el retriever.
    """

    def embed_documents(self, textos: list[str]) -> np.ndarray:
        """Devuelve una matriz (len(textos), dim) para textos del corpus."""
        ...

    def embed_query(self, texto: str) -> np.ndarray:
        """Devuelve una matriz (1, dim) para una consulta del usuario."""
        ...


# nomic-embed-text es un modelo asimétrico: se entrenó con estos prefijos para
# distinguir el rol de cada texto, así que es su forma de uso documentada. Mejora
# el ordenamiento, pero no separa por completo las consultas del dominio de las
# ajenas: ver la nota sobre el umbral en `retriever.CosineRetriever`.
PREFIJOS = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}


class OllamaEmbeddings:
    """Proveedor de embeddings respaldado por un servidor Ollama local."""

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self._client = ollama.Client(host=host)
        # La clave se busca sin el tag (`nomic-embed-text:latest` -> `nomic-embed-text`).
        self._prefijo_doc, self._prefijo_query = PREFIJOS.get(model.split(":")[0], ("", ""))

    def _embed(self, textos: list[str]) -> np.ndarray:
        if not textos:
            return np.empty((0, 0), dtype=np.float32)
        try:
            respuesta = self._client.embed(model=self.model, input=textos)
        except Exception as exc:  # ollama envuelve errores de red y de modelo faltante
            raise EmbeddingError(
                f"No se pudieron generar embeddings con el modelo '{self.model}'. "
                f"¿Está Ollama corriendo y el modelo descargado "
                f"(`ollama pull {self.model}`)? Detalle: {exc}"
            ) from exc
        return np.asarray(respuesta["embeddings"], dtype=np.float32)

    def embed_documents(self, textos: list[str]) -> np.ndarray:
        return self._embed([f"{self._prefijo_doc}{t}" for t in textos])

    def embed_query(self, texto: str) -> np.ndarray:
        return self._embed([f"{self._prefijo_query}{texto}"])


# Se incluye en la huella del cache. Hay que subirlo cuando cambie *cómo* se
# embeben los textos sin que cambien los textos en sí (por ejemplo, al tocar los
# prefijos de arriba): si no, el cache viejo seguiría pareciendo válido.
_CACHE_VERSION = 2


def _hash_documentos(textos: list[str], model: str) -> str:
    """Huella del corpus + modelo + versión: si cambia alguno, el cache se invalida."""
    h = hashlib.sha256(f"{_CACHE_VERSION}:{model}".encode())
    for texto in textos:
        h.update(texto.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def embed_documentos_con_cache(
    textos: list[str],
    provider: EmbeddingProvider,
    cache_dir: Path,
    model_name: str,
) -> np.ndarray:
    """Embebe `textos`, reutilizando el cache en disco si sigue siendo válido.

    El cache guarda la huella del corpus junto a los vectores; si no coincide se
    ignora y se recalcula, así que nunca se sirven embeddings desactualizados.
    """
    huella = _hash_documentos(textos, model_name)
    cache_path = cache_dir / "embeddings.npz"

    if cache_path.exists():
        try:
            guardado = np.load(cache_path, allow_pickle=False)
            if str(guardado["huella"]) == huella:
                logger.info("Embeddings cargados desde cache (%s documentos)", len(textos))
                return guardado["vectores"]
            logger.info("El cache de embeddings quedó viejo, se recalcula")
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Cache de embeddings ilegible (%s), se recalcula", exc)

    logger.info("Generando embeddings de %s documentos con '%s'...", len(textos), model_name)
    vectores = provider.embed_documents(textos)

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, vectores=vectores, huella=np.array(huella))
    except OSError as exc:
        # No poder cachear no es fatal: el arranque simplemente será más lento.
        logger.warning("No se pudo escribir el cache de embeddings: %s", exc)

    return vectores
