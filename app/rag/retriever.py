"""Búsqueda semántica sobre los documentos del negocio.

Reemplaza el matcheo exacto de la versión original, que preguntaba si la consulta
completa era idéntica a una celda del CSV (y por lo tanto nunca acertaba con
preguntas en lenguaje natural).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.rag.embeddings import EmbeddingProvider, embed_documentos_con_cache
from app.rag.loader import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Resultado:
    """Un documento recuperado junto a su similitud con la consulta."""

    document: Document
    score: float


class Retriever(Protocol):
    """Contrato de recuperación.

    Aísla al resto de la app de cómo se busca. Cambiar esta implementación por
    una respaldada por un vector store no requiere tocar las rutas ni el prompt.
    """

    def search(self, query: str, top_k: int) -> list[Resultado]:
        """Devuelve hasta `top_k` documentos relevantes, del más al menos relevante."""
        ...


def _normalizar(matriz: np.ndarray) -> np.ndarray:
    """Normaliza cada fila a norma 1, para que el producto punto sea la similitud coseno."""
    normas = np.linalg.norm(matriz, axis=1, keepdims=True)
    # Evita dividir por cero ante un vector nulo (documento vacío o error del modelo).
    normas[normas == 0] = 1.0
    return matriz / normas


class CosineRetriever:
    """Búsqueda por similitud coseno con un índice denso en memoria.

    Con ~40 documentos, un vector store agregaría una dependencia pesada sin
    ganancia medible: el índice entra en unos pocos kilobytes y la búsqueda es
    un producto matriz-vector. Si el catálogo creciera a decenas de miles de
    filas, la sustitución natural es implementar `Retriever` sobre FAISS o
    pgvector, sin cambios en el resto de la aplicación.

    Sobre el umbral: medido con nomic-embed-text sobre este catálogo, las
    consultas del dominio puntúan entre 0.62 y 0.79, y las ajenas entre 0.55 y
    0.65. Los rangos se superponen, así que **ningún umbral separa por completo
    un tema del otro** y sería deshonesto presentarlo como un filtro de
    relevancia. Lo que hace es recortar la cola de coincidencias flojas dentro
    de una misma consulta, para que el modelo reciba dos documentos buenos en
    vez de cinco mediocres. Rechazar preguntas fuera de tema es responsabilidad
    del prompt de sistema, que es donde se decide con el texto a la vista.
    """

    def __init__(
        self,
        documentos: list[Document],
        vectores: np.ndarray,
        provider: EmbeddingProvider,
        threshold: float = 0.45,
    ) -> None:
        if len(documentos) != len(vectores):
            raise ValueError(
                f"Cantidad de documentos ({len(documentos)}) y de vectores "
                f"({len(vectores)}) no coincide."
            )
        self.documentos = documentos
        self._matriz = _normalizar(np.asarray(vectores, dtype=np.float32))
        self._provider = provider
        self.threshold = threshold

    @classmethod
    def desde_documentos(
        cls,
        documentos: list[Document],
        provider: EmbeddingProvider,
        cache_dir: Path,
        model_name: str,
        threshold: float = 0.45,
    ) -> "CosineRetriever":
        """Construye el índice embebiendo los documentos (con cache en disco)."""
        textos = [doc.text for doc in documentos]
        vectores = embed_documentos_con_cache(textos, provider, cache_dir, model_name)
        return cls(documentos, vectores, provider, threshold)

    def search(self, query: str, top_k: int = 5) -> list[Resultado]:
        """Recupera los documentos más parecidos a `query` por encima del umbral.

        Devolver una lista vacía es un resultado válido y esperado: significa que
        la pregunta no es sobre el negocio, y el prompt usa esa señal para que el
        modelo diga que no tiene el dato en vez de inventarlo.
        """
        if not query.strip() or not self.documentos:
            return []

        query_vec = self._provider.embed_query(query)
        if query_vec.size == 0:
            return []
        query_vec = _normalizar(query_vec)[0]

        similitudes = self._matriz @ query_vec

        # argpartition evita ordenar todo el índice; con 40 docs da igual, pero
        # mantiene el costo en O(n) si el catálogo crece.
        k = min(top_k, len(similitudes))
        candidatos = np.argpartition(-similitudes, k - 1)[:k]
        candidatos = candidatos[np.argsort(-similitudes[candidatos])]

        resultados = [
            Resultado(document=self.documentos[i], score=float(similitudes[i]))
            for i in candidatos
            if similitudes[i] >= self.threshold
        ]
        logger.debug(
            "Consulta %r -> %s documentos sobre el umbral %.2f",
            query,
            len(resultados),
            self.threshold,
        )
        return resultados
