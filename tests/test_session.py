"""Tests del historial por sesión."""

from app.session import SessionStore


def test_sesiones_no_comparten_historial():
    """El bug original: una lista global mezclaba las conversaciones de todos."""
    store = SessionStore()
    a = store.obtener_o_crear(None)
    b = store.obtener_o_crear(None)

    store.agregar(a.id, "user", "hola desde A")

    assert a.id != b.id
    assert store.historial(b.id) == []
    assert len(store.historial(a.id)) == 1


def test_reutiliza_la_sesion_por_id():
    store = SessionStore()
    sesion = store.obtener_o_crear(None)
    store.agregar(sesion.id, "user", "primera")

    de_nuevo = store.obtener_o_crear(sesion.id)

    assert de_nuevo.id == sesion.id
    assert len(store.historial(sesion.id)) == 1


def test_id_desconocido_crea_sesion_nueva():
    store = SessionStore()
    sesion = store.obtener_o_crear("no-existe")
    assert sesion.id != "no-existe"


def test_historial_se_recorta_a_los_turnos_configurados():
    store = SessionStore(max_turns=2)
    sesion = store.obtener_o_crear(None)

    for i in range(5):
        store.agregar(sesion.id, "user", f"pregunta {i}")
        store.agregar(sesion.id, "assistant", f"respuesta {i}")

    historial = store.historial(sesion.id)
    assert len(historial) == 4  # 2 turnos = 4 mensajes
    assert historial[0]["content"] == "pregunta 3"


def test_sesiones_vencidas_se_descartan():
    store = SessionStore(ttl_seconds=0)
    vieja = store.obtener_o_crear(None)
    store.agregar(vieja.id, "user", "hola")

    store.obtener_o_crear(None)  # el purgado corre al pedir una sesión

    assert len(store) == 1


def test_historial_es_una_copia():
    """Mutar lo devuelto no debe alterar el estado interno."""
    store = SessionStore()
    sesion = store.obtener_o_crear(None)
    store.agregar(sesion.id, "user", "hola")

    store.historial(sesion.id).append({"role": "user", "content": "inyectado"})

    assert len(store.historial(sesion.id)) == 1
