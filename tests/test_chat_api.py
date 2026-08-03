"""Tests de contrato de la API.

La app se arma sin ejecutar el lifespan: el estado (retriever, LLM, sesiones) se
inyecta a mano con dobles, así los tests corren sin Ollama y sin leer `data/`.
"""

import json
import re
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.ollama_client import LLMError
from app.main import app
from app.rag.embeddings import EmbeddingError
from app.rag.retriever import CosineRetriever, Resultado
from app.session import SessionStore


class FakeLLM:
    """LLM que devuelve un texto fijo palabra por palabra."""

    def __init__(self, respuesta: str = "El Malbec sale $1200.", modelos=None, error=None):
        self.respuesta = respuesta
        self.modelos = modelos if modelos is not None else ["llama3.2:3b", "nomic-embed-text"]
        self.error = error
        self.ultimos_mensajes: list[dict[str, str]] | None = None

    async def stream(self, messages) -> AsyncIterator[str]:
        self.ultimos_mensajes = messages
        if self.error:
            raise self.error
        for palabra in self.respuesta.split(" "):
            yield palabra + " "

    async def modelos_disponibles(self) -> list[str]:
        if self.error:
            raise self.error
        return self.modelos


class FakeRetriever:
    def __init__(self, resultados=None, error=None):
        self.documentos = []
        self._resultados = resultados or []
        self._error = error

    def search(self, query: str, top_k: int = 5) -> list[Resultado]:
        if self._error:
            raise self._error
        return self._resultados


def eventos(respuesta) -> list[tuple[str, dict]]:
    """Parsea el cuerpo SSE en una lista de (evento, datos)."""
    parseados = []
    for bloque in respuesta.text.split("\n\n"):
        if not bloque.strip():
            continue
        nombre, datos = "message", ""
        for linea in bloque.split("\n"):
            if linea.startswith("event:"):
                nombre = linea[6:].strip()
            elif linea.startswith("data:"):
                datos += linea[5:].strip()
        parseados.append((nombre, json.loads(datos)))
    return parseados


def texto_de(eventos_parseados: list[tuple[str, dict]]) -> str:
    return "".join(d["text"] for e, d in eventos_parseados if e == "token")


@pytest.fixture
def montar(documentos, fake_embeddings):
    """Devuelve un factory que arma el TestClient con el estado que pida el test."""

    def _montar(retriever="real", llm=None):
        if retriever == "real":
            vectores = fake_embeddings.embed_documents([d.text for d in documentos])
            retriever = CosineRetriever(documentos, vectores, fake_embeddings, threshold=0.2)

        app.state.settings = Settings()
        app.state.sessions = SessionStore(max_turns=10, ttl_seconds=3600)
        app.state.llm = llm or FakeLLM()
        app.state.retriever = retriever
        return TestClient(app)

    return _montar


# --- Camino feliz ---


def test_chat_devuelve_los_eventos_en_orden(montar):
    cliente = montar()
    r = cliente.post("/api/chat", json={"content": "¿cuánto sale el malbec?"})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    nombres = [e for e, _ in eventos(r)]
    assert nombres[0] == "session"
    assert nombres[1] == "sources"
    assert nombres[-1] == "done"
    assert "token" in nombres


def test_chat_devuelve_la_respuesta_del_modelo(montar):
    cliente = montar()
    r = cliente.post("/api/chat", json={"content": "¿cuánto sale el malbec?"})

    assert "El Malbec sale $1200." in texto_de(eventos(r))


def test_chat_cita_las_fuentes_recuperadas(montar):
    cliente = montar()
    r = cliente.post("/api/chat", json={"content": "¿cuánto sale el malbec?"})

    fuentes = next(d["sources"] for e, d in eventos(r) if e == "sources")
    assert fuentes
    assert "Malbec" in fuentes[0]["text"]
    assert 0.0 <= fuentes[0]["score"] <= 1.0


def test_el_prompt_incluye_el_contexto_recuperado(montar):
    llm = FakeLLM()
    cliente = montar(llm=llm)
    cliente.post("/api/chat", json={"content": "¿cuánto sale el malbec?"})

    assert llm.ultimos_mensajes[0]["role"] == "system"
    # El contexto va en su propio mensaje de sistema, justo antes de la consulta.
    assert llm.ultimos_mensajes[-2]["role"] == "system"
    assert "Malbec" in llm.ultimos_mensajes[-2]["content"]


def test_el_contexto_no_lleva_rotulos_en_mayuscula(montar):
    """Regresión: un rótulo llamativo en el contexto termina copiado en la respuesta.

    Con "DATOS DE LA VINOTECA:" encabezando el bloque, llama3.2:3b arrancaba
    algunas respuestas repitiéndolo literalmente. Sin nada que se lea como
    etiqueta, no hay qué imitar.
    """
    llm = FakeLLM()
    cliente = montar(llm=llm)
    cliente.post("/api/chat", json={"content": "¿qué vinos tintos tienen?"})

    contexto = llm.ultimos_mensajes[-2]["content"]
    palabras = re.findall(r"\b[A-ZÁÉÍÓÚÑ]{3,}\b", contexto)
    assert not palabras, f"El contexto tiene rótulos en mayúscula: {palabras}"


def test_el_mensaje_del_usuario_no_lleva_andamiaje(montar):
    """Regresión: el contexto no debe contaminar el turno del usuario.

    Cuando iba embebido ahí, el historial quedaba con mensajes de usuario de dos
    formas distintas y el modelo terminaba copiando el encabezado del prompt en
    su respuesta e inventando precios.
    """
    llm = FakeLLM()
    cliente = montar(llm=llm)
    cliente.post("/api/chat", json={"content": "¿cuánto sale el malbec?"})

    ultimo = llm.ultimos_mensajes[-1]
    assert ultimo["role"] == "user"
    assert ultimo["content"] == "¿cuánto sale el malbec?"


def test_el_historial_guarda_las_preguntas_crudas(montar):
    """Los turnos viejos y el actual tienen que tener el mismo formato."""
    llm = FakeLLM()
    cliente = montar(llm=llm)

    r = cliente.post("/api/chat", json={"content": "¿tienen malbec?"})
    sid = next(d["session_id"] for e, d in eventos(r) if e == "session")
    cliente.post("/api/chat", json={"content": "¿y cuánto sale?", "session_id": sid})

    usuarios = [m["content"] for m in llm.ultimos_mensajes if m["role"] == "user"]
    assert usuarios == ["¿tienen malbec?", "¿y cuánto sale?"]


# --- Sesiones ---


def test_sesiones_distintas_no_comparten_historial(montar):
    llm = FakeLLM()
    cliente = montar(llm=llm)

    r1 = cliente.post("/api/chat", json={"content": "primera"})
    sid1 = next(d["session_id"] for e, d in eventos(r1) if e == "session")

    r2 = cliente.post("/api/chat", json={"content": "segunda"})
    sid2 = next(d["session_id"] for e, d in eventos(r2) if e == "session")

    assert sid1 != sid2
    # El prompt de la segunda sesión no debe arrastrar la pregunta de la primera.
    historial = [m["content"] for m in llm.ultimos_mensajes]
    assert not any("primera" in c for c in historial)


def test_reenviar_el_session_id_mantiene_el_historial(montar):
    llm = FakeLLM()
    cliente = montar(llm=llm)

    r1 = cliente.post("/api/chat", json={"content": "¿tienen malbec?"})
    sid = next(d["session_id"] for e, d in eventos(r1) if e == "session")

    cliente.post("/api/chat", json={"content": "¿y cuánto sale?", "session_id": sid})

    historial = [m["content"] for m in llm.ultimos_mensajes]
    assert any("¿tienen malbec?" in c for c in historial)


def test_el_prompt_de_sistema_no_se_pierde_con_el_historial(montar):
    """El truncado por antigüedad no puede comerse las instrucciones."""
    llm = FakeLLM()
    cliente = montar(llm=llm)

    r = cliente.post("/api/chat", json={"content": "hola"})
    sid = next(d["session_id"] for e, d in eventos(r) if e == "session")
    for i in range(25):
        cliente.post("/api/chat", json={"content": f"consulta {i}", "session_id": sid})

    assert llm.ultimos_mensajes[0]["role"] == "system"
    assert "Enotek Vinos" in llm.ultimos_mensajes[0]["content"]


# --- Errores ---


def test_sin_indice_devuelve_503(montar):
    cliente = montar(retriever=None)
    r = cliente.post("/api/chat", json={"content": "hola"})

    assert r.status_code == 503
    assert "índice" in r.json()["detail"]


def test_ollama_caido_emite_evento_de_error(montar):
    llm = FakeLLM(error=LLMError("No se pudo contactar a Ollama en http://127.0.0.1:11434"))
    cliente = montar(llm=llm)

    r = cliente.post("/api/chat", json={"content": "hola"})
    parseados = eventos(r)

    assert r.status_code == 200  # el stream ya empezó: el error viaja como evento
    mensaje = next(d["message"] for e, d in parseados if e == "error")
    assert "Ollama" in mensaje
    assert "done" not in [e for e, _ in parseados]


def test_falla_de_embeddings_emite_evento_de_error(montar):
    cliente = montar(retriever=FakeRetriever(error=EmbeddingError("modelo no descargado")))
    r = cliente.post("/api/chat", json={"content": "hola"})

    assert "modelo no descargado" in next(d["message"] for e, d in eventos(r) if e == "error")


@pytest.mark.parametrize("payload", [{}, {"content": ""}, {"content": "x" * 2001}])
def test_payload_invalido_devuelve_422(montar, payload):
    assert montar().post("/api/chat", json=payload).status_code == 422


# --- Health ---


def test_health_ok(montar):
    r = montar().get("/api/health")
    datos = r.json()

    assert datos["status"] == "ok"
    assert datos["ollama"] is True
    assert datos["chat_model_disponible"] is True
    assert datos["documentos_indexados"] == 3


def test_health_reporta_ollama_caido(montar):
    cliente = montar(llm=FakeLLM(error=LLMError("connection refused")))
    datos = cliente.get("/api/health").json()

    assert datos["status"] == "degraded"
    assert datos["ollama"] is False
    assert "connection refused" in datos["detalle"]


def test_health_reporta_modelo_faltante(montar):
    cliente = montar(llm=FakeLLM(modelos=["otro-modelo:1b"]))
    datos = cliente.get("/api/health").json()

    assert datos["status"] == "degraded"
    assert datos["ollama"] is True
    assert datos["chat_model_disponible"] is False
    assert "ollama pull" in datos["detalle"]


def test_health_reporta_indice_vacio(montar):
    cliente = montar(retriever=FakeRetriever())
    datos = cliente.get("/api/health").json()

    assert datos["status"] == "degraded"
    assert datos["documentos_indexados"] == 0


# --- Frontend ---


def test_la_raiz_sirve_el_frontend(montar):
    r = montar().get("/")

    assert r.status_code == 200
    assert "Enotek Vinos" in r.text
