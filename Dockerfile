# --- Etapa de build: instala dependencias en un virtualenv aislado ---
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Se copia sólo el manifiesto primero: si no cambian las dependencias,
# Docker reutiliza esta capa aunque cambie el código.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

# --- Etapa final: sólo el runtime ---
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv

COPY app ./app
COPY static ./static
COPY data ./data

# Usuario sin privilegios. El directorio de cache tiene que ser suyo para que
# pueda escribir los embeddings.
RUN useradd --create-home --uid 1000 vinoteca \
    && mkdir -p /srv/.cache \
    && chown -R vinoteca:vinoteca /srv

USER vinoteca

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
