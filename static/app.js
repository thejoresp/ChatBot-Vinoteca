/**
 * Cliente del asistente de Enotek Vinos.
 *
 * Consume el endpoint POST /api/chat, que responde en Server-Sent Events.
 * Se parsea el stream a mano porque EventSource sólo sabe hacer GET.
 */

const $ = (id) => document.getElementById(id);

const els = {
    messages: $("messages"),
    form: $("composer"),
    input: $("message"),
    send: $("send"),
    banner: $("banner"),
    statusDot: $("status-dot"),
    statusText: $("status-text"),
    suggestions: $("suggestions"),
};

let sessionId = null;
let enviando = false;

// --- Render ---------------------------------------------------------------

const escapar = (texto) =>
    texto.replace(/[&<>"']/g, (c) => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
    ));

/**
 * Markdown mínimo: negritas, itálicas, código y listas con guiones.
 * Suficiente para lo que devuelve el modelo, y sin dependencias externas
 * (el proyecto tiene que funcionar sin conexión a internet).
 */
function renderMarkdown(texto) {
    const inline = (t) =>
        escapar(t)
            .replace(/`([^`]+)`/g, "<code>$1</code>")
            .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
            .replace(/(^|\W)\*([^*\n]+)\*/g, "$1<em>$2</em>");

    const html = [];
    let lista = null;

    for (const linea of texto.split("\n")) {
        const item = linea.match(/^\s*[-*•]\s+(.*)$/);
        if (item) {
            lista ??= [];
            lista.push(`<li>${inline(item[1])}</li>`);
            continue;
        }
        if (lista) {
            html.push(`<ul>${lista.join("")}</ul>`);
            lista = null;
        }
        if (linea.trim()) html.push(`<p>${inline(linea)}</p>`);
    }
    if (lista) html.push(`<ul>${lista.join("")}</ul>`);

    return html.join("");
}

function limpiarEstadoInicial() {
    els.messages.querySelector(".empty")?.remove();
}

function agregarMensaje(tipo, contenido = "") {
    limpiarEstadoInicial();
    const div = document.createElement("div");
    div.className = `msg msg--${tipo}`;
    if (contenido) div.textContent = contenido;
    els.messages.appendChild(div);
    scrollAlFinal();
    return div;
}

function mostrarEscribiendo(el) {
    el.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
}

function renderFuentes(el, fuentes) {
    if (!fuentes.length) return;
    const detalles = document.createElement("details");
    detalles.className = "sources";
    const items = fuentes
        .map((f) => `<li>${escapar(f.text)} <span class="score">(${f.score.toFixed(2)})</span></li>`)
        .join("");
    detalles.innerHTML =
        `<summary>${fuentes.length} dato${fuentes.length > 1 ? "s" : ""} de la vinoteca` +
        `</summary><ul>${items}</ul>`;
    el.appendChild(detalles);
}

function scrollAlFinal() {
    els.messages.scrollTop = els.messages.scrollHeight;
}

function mostrarBanner(html) {
    els.banner.innerHTML = html;
    els.banner.hidden = false;
}

function ocultarBanner() {
    els.banner.hidden = true;
}

function bloquearEntrada(bloqueada) {
    enviando = bloqueada;
    els.input.disabled = bloqueada;
    els.send.disabled = bloqueada;
    els.suggestions.querySelectorAll(".chip").forEach((c) => (c.disabled = bloqueada));
    if (!bloqueada) els.input.focus();
}

// --- Estado del servicio --------------------------------------------------

function setEstado(clase, texto) {
    els.statusDot.className = `status__dot status__dot--${clase}`;
    els.statusText.textContent = texto;
}

async function verificarEstado() {
    try {
        const r = await fetch("/api/health");
        const data = await r.json();

        if (data.status === "ok") {
            setEstado("ok", data.chat_model);
            ocultarBanner();
            return;
        }

        setEstado("warn", "Configuración incompleta");
        if (!data.ollama) {
            mostrarBanner(
                "<strong>Ollama no responde.</strong> Arrancá el servidor con " +
                "<code>ollama serve</code> y recargá la página."
            );
        } else if (!data.chat_model_disponible) {
            mostrarBanner(
                `<strong>Falta el modelo <code>${escapar(data.chat_model)}</code>.</strong> ` +
                `Descargalo con <code>ollama pull ${escapar(data.chat_model)}</code>.`
            );
        } else {
            mostrarBanner(`<strong>Servicio degradado.</strong> ${escapar(data.detalle || "")}`);
        }
    } catch {
        setEstado("error", "Sin conexión");
        mostrarBanner(
            "<strong>No se pudo contactar al servidor.</strong> " +
            "Verificá que esté corriendo <code>uvicorn app.main:app</code>."
        );
    }
}

// --- Envío ----------------------------------------------------------------

async function enviar(texto) {
    if (enviando || !texto.trim()) return;

    agregarMensaje("user", texto);
    els.input.value = "";
    bloquearEntrada(true);

    const burbuja = agregarMensaje("bot");
    mostrarEscribiendo(burbuja);

    let respuesta = "";
    let fuentes = [];
    let huboError = false;

    try {
        const r = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: texto, session_id: sessionId }),
        });

        if (!r.ok) {
            const detalle = await r.json().catch(() => ({}));
            throw new Error(detalle.detail || `El servidor respondió ${r.status}.`);
        }

        for await (const { evento, datos } of leerSSE(r.body)) {
            switch (evento) {
                case "session":
                    sessionId = datos.session_id;
                    break;
                case "sources":
                    fuentes = datos.sources;
                    break;
                case "token":
                    respuesta += datos.text;
                    burbuja.innerHTML = renderMarkdown(respuesta);
                    scrollAlFinal();
                    break;
                case "error":
                    huboError = true;
                    burbuja.className = "msg msg--error";
                    burbuja.textContent = datos.message;
                    break;
            }
        }

        if (!huboError) {
            if (!respuesta.trim()) {
                burbuja.className = "msg msg--error";
                burbuja.textContent = "El modelo no devolvió ninguna respuesta. Probá de nuevo.";
            } else {
                burbuja.innerHTML = renderMarkdown(respuesta);
                renderFuentes(burbuja, fuentes);
            }
        }
    } catch (err) {
        burbuja.className = "msg msg--error";
        burbuja.textContent = `No se pudo obtener la respuesta: ${err.message}`;
        verificarEstado();
    } finally {
        // Pase lo que pase, la entrada vuelve a estar disponible.
        bloquearEntrada(false);
        scrollAlFinal();
    }
}

/** Parsea un stream SSE y va emitiendo `{ evento, datos }`. */
async function* leerSSE(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Los eventos SSE se separan con una línea en blanco.
        const bloques = buffer.split("\n\n");
        buffer = bloques.pop() ?? "";

        for (const bloque of bloques) {
            let evento = "message";
            let raw = "";
            for (const linea of bloque.split("\n")) {
                if (linea.startsWith("event:")) evento = linea.slice(6).trim();
                else if (linea.startsWith("data:")) raw += linea.slice(5).trim();
            }
            if (!raw) continue;
            try {
                yield { evento, datos: JSON.parse(raw) };
            } catch {
                // Evento malformado: se ignora en vez de cortar el stream.
            }
        }
    }
}

// --- Arranque -------------------------------------------------------------

els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    enviar(els.input.value);
});

els.suggestions.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip && !enviando) enviar(chip.textContent.trim());
});

els.messages.innerHTML =
    '<p class="empty">Hola 👋 Soy el asistente de Enotek Vinos.<br>' +
    "Preguntame por precios, sucursales u horarios.</p>";

verificarEstado();
els.input.focus();
