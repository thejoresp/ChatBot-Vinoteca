"""Tests de la búsqueda semántica."""

import numpy as np
import pytest

from app.rag.embeddings import embed_documentos_con_cache
from app.rag.retriever import CosineRetriever


@pytest.fixture
def retriever(documentos, fake_embeddings):
    vectores = fake_embeddings.embed_documents([d.text for d in documentos])
    return CosineRetriever(documentos, vectores, fake_embeddings, threshold=0.2)


def test_pregunta_en_lenguaje_natural_encuentra_el_producto(retriever):
    """Regresión del bug original.

    La implementación vieja comparaba la consulta completa contra cada celda del
    CSV, así que "¿cuánto sale el malbec?" no matcheaba nunca y el modelo
    respondía sin datos. Este test falla contra aquella versión.
    """
    resultados = retriever.search("¿cuánto sale el malbec?", top_k=3)

    assert resultados, "La consulta no recuperó ningún documento"
    assert "Malbec" in resultados[0].document.text


def test_pregunta_por_sucursal_recupera_la_ubicacion(retriever):
    resultados = retriever.search("¿dónde queda la sucursal de Córdoba?", top_k=3)

    assert resultados
    assert resultados[0].document.source == "ubicaciones"
    assert "Córdoba" in resultados[0].document.text


def test_resultados_ordenados_por_score(retriever):
    resultados = retriever.search("malbec vino tinto precio", top_k=3)
    scores = [r.score for r in resultados]
    assert scores == sorted(scores, reverse=True)


def test_consulta_irrelevante_no_devuelve_nada(documentos, fake_embeddings):
    """Sin resultados, el prompt le pide al modelo que admita no tener el dato."""
    vectores = fake_embeddings.embed_documents([d.text for d in documentos])
    estricto = CosineRetriever(documentos, vectores, fake_embeddings, threshold=0.9)

    assert estricto.search("cuál es la capital de Francia", top_k=3) == []


def test_consulta_vacia_no_llama_al_modelo(retriever, fake_embeddings):
    fake_embeddings.llamadas.clear()
    assert retriever.search("   ", top_k=3) == []
    assert fake_embeddings.llamadas == []


def test_top_k_limita_la_cantidad(retriever):
    assert len(retriever.search("vino cerveza sucursal precio", top_k=2)) <= 2


def test_desalineacion_documentos_vectores_falla_al_construir(documentos, fake_embeddings):
    vectores = fake_embeddings.embed_documents([documentos[0].text])
    with pytest.raises(ValueError, match="no coincide"):
        CosineRetriever(documentos, vectores, fake_embeddings)


def test_vector_nulo_no_rompe_la_normalizacion(fake_embeddings, documentos):
    vectores = np.zeros((len(documentos), 5), dtype=np.float32)
    retriever = CosineRetriever(documentos, vectores, fake_embeddings, threshold=-1.0)
    assert np.isfinite(retriever._matriz).all()


# --- Cache de embeddings ---


def test_cache_evita_recalcular(tmp_path, fake_embeddings):
    textos = ["uno", "dos"]

    primero = embed_documentos_con_cache(textos, fake_embeddings, tmp_path, "modelo-test")
    assert len(fake_embeddings.llamadas) == 1

    segundo = embed_documentos_con_cache(textos, fake_embeddings, tmp_path, "modelo-test")
    assert len(fake_embeddings.llamadas) == 1, "Se recalculó teniendo cache válido"
    np.testing.assert_array_equal(primero, segundo)


def test_cache_se_invalida_si_cambian_los_documentos(tmp_path, fake_embeddings):
    embed_documentos_con_cache(["uno"], fake_embeddings, tmp_path, "modelo-test")
    embed_documentos_con_cache(["uno", "dos"], fake_embeddings, tmp_path, "modelo-test")

    assert len(fake_embeddings.llamadas) == 2


def test_cache_se_invalida_si_cambia_el_modelo(tmp_path, fake_embeddings):
    embed_documentos_con_cache(["uno"], fake_embeddings, tmp_path, "modelo-a")
    embed_documentos_con_cache(["uno"], fake_embeddings, tmp_path, "modelo-b")

    assert len(fake_embeddings.llamadas) == 2


def test_cache_corrupto_se_recalcula(tmp_path, fake_embeddings):
    (tmp_path / "embeddings.npz").write_bytes(b"basura")

    vectores = embed_documentos_con_cache(["uno"], fake_embeddings, tmp_path, "modelo-test")

    assert vectores.shape[0] == 1
    assert len(fake_embeddings.llamadas) == 1
