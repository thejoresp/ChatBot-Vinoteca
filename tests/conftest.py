"""Fixtures compartidas.

Ningún test requiere Ollama corriendo: tanto los embeddings como el LLM se
reemplazan por dobles deterministas.
"""

from typing import ClassVar

import numpy as np
import pytest

from app.rag.loader import Document


@pytest.fixture
def data_dir(tmp_path):
    """Un directorio de datos mínimo pero con la misma forma que `data/`."""
    (tmp_path / "precios.csv").write_text(
        "Categoría,Producto,Precio\n"
        "Vino Tinto,Malbec (Bodega Norton),1200\n"
        "Vino Tinto,Cabernet Sauvignon (Trapiche),1500\n"
        "Vino Blanco,Chardonnay (Luigi Bosca),1600\n"
        "Cerveza,IPA Artesanal,900\n",
        encoding="utf-8",
    )
    (tmp_path / "ubicaciones.csv").write_text(
        "Ciudad,Sucursal,Dirección,Horarios\n"
        'Buenos Aires,Sucursal Palermo,"Av. Santa Fe 1234",'
        '"Lunes a Sábado: 10:00 a 20:00, Domingo: Cerrado"\n'
        'Córdoba,Sucursal Centro,"Av. Colón 789",'
        '"Lunes a Sábado: 09:00 a 19:00, Domingo: Cerrado"\n',
        encoding="utf-8",
    )
    return tmp_path


class FakeEmbeddings:
    """Proveedor de embeddings determinista basado en bolsa de palabras.

    No pretende imitar la calidad semántica de un modelo real: sólo garantiza
    que textos con vocabulario en común queden cerca, que es la propiedad que
    los tests del retriever necesitan verificar.
    """

    VOCABULARIO: ClassVar[list[str]] = [
        "malbec", "cabernet", "chardonnay", "ipa", "precio", "pesos",
        "vino", "tinto", "blanco", "cerveza", "sucursal", "palermo",
        "centro", "córdoba", "buenos", "aires", "horarios", "dirección",
        "abren", "sale", "cuánto", "dónde", "queda",
    ]  # fmt: skip

    def __init__(self):
        self.llamadas: list[list[str]] = []

    def _embed(self, textos: list[str]) -> np.ndarray:
        self.llamadas.append(textos)
        if not textos:
            return np.empty((0, 0), dtype=np.float32)
        vectores = []
        for texto in textos:
            bajo = texto.lower()
            fila = [1.0 if palabra in bajo else 0.0 for palabra in self.VOCABULARIO]
            # Componente constante chica: evita el vector nulo si no hay coincidencias.
            fila.append(0.05)
            vectores.append(fila)
        return np.asarray(vectores, dtype=np.float32)

    def embed_documents(self, textos: list[str]) -> np.ndarray:
        return self._embed(textos)

    def embed_query(self, texto: str) -> np.ndarray:
        return self._embed([texto])


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.fixture
def documentos():
    return [
        Document(text="Producto: Malbec. Categoría: Vino Tinto. Precio: $1200.", source="precios"),
        Document(
            text="Producto: IPA Artesanal. Categoría: Cerveza. Precio: $900.", source="precios"
        ),
        Document(
            text="Sucursal: Sucursal Centro, en la ciudad de Córdoba. Dirección: Av. Colón 789.",
            source="ubicaciones",
        ),
    ]
