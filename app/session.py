"""Historial de conversación por sesión.

La versión original guardaba los mensajes en una lista global del módulo: todos
los usuarios compartían la misma conversación y el truncado podía comerse el
prompt de sistema. Acá cada sesión tiene su historial, el prompt de sistema vive
fuera de él, y las sesiones inactivas se descartan.
"""

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Session:
    """Conversación de un usuario."""

    id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)


class SessionStore:
    """Almacén de sesiones en memoria con TTL y límite de historial.

    En memoria alcanza para una demo local monoproceso. Para varios workers o
    persistencia entre reinicios, la sustitución natural es Redis manteniendo
    esta misma interfaz.
    """

    def __init__(self, max_turns: int = 10, ttl_seconds: int = 3600) -> None:
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def _purgar(self, ahora: float) -> None:
        """Descarta sesiones inactivas. Se llama con el lock tomado."""
        vencidas = [
            sid for sid, s in self._sessions.items() if ahora - s.last_seen > self.ttl_seconds
        ]
        for sid in vencidas:
            del self._sessions[sid]

    def obtener_o_crear(self, session_id: str | None) -> Session:
        """Devuelve la sesión pedida, o crea una nueva si no existe o venció."""
        ahora = time.monotonic()
        with self._lock:
            self._purgar(ahora)
            sesion = self._sessions.get(session_id) if session_id else None
            if sesion is None:
                sesion = Session(id=str(uuid.uuid4()))
                self._sessions[sesion.id] = sesion
            sesion.last_seen = ahora
            return sesion

    def agregar(self, session_id: str, role: str, content: str) -> None:
        """Agrega un mensaje al historial, recortando los turnos más viejos."""
        with self._lock:
            sesion = self._sessions.get(session_id)
            if sesion is None:
                return
            sesion.messages.append({"role": role, "content": content})
            # Un turno son dos mensajes (usuario + asistente).
            limite = self.max_turns * 2
            if len(sesion.messages) > limite:
                sesion.messages = sesion.messages[-limite:]
            sesion.last_seen = time.monotonic()

    def historial(self, session_id: str) -> list[dict[str, str]]:
        """Copia del historial de la sesión, o lista vacía si no existe."""
        with self._lock:
            sesion = self._sessions.get(session_id)
            return list(sesion.messages) if sesion else []

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
