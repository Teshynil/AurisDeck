FROM python:3.12-slim-bookworm

ARG INSTALL_DIARIZATION=false

ENV PATH="/opt/aurisdeck/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME="/data/models" \
    AURISDECK_DATA_DIR="/data"

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        gosu \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/aurisdeck

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN if [ "$INSTALL_DIARIZATION" = "true" ]; then \
        uv sync --frozen --no-dev --extra diarization --no-cache; \
    else \
        uv sync --frozen --no-dev --no-cache; \
    fi

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["aurisdeck", "serve", "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/data"]
