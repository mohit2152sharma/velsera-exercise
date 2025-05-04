# NOTE: This image is not going to work, using pytorch images will be better
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# NOTE: when using --no-install-recommends and rm var/list it throws an error
RUN apt-get update \
    && apt-get -y install libpq-dev gcc

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-install-workspace

ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Add a layer for building celery image
FROM python:3.13-slim-bookworm as finetuner

RUN useradd -m app
COPY --from=builder --chown=app:app /app /app
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
USER app

CMD ["uv run python -m velsera.finetuning.main"]

FROM python:3.13-slim-bookworm as webapp

COPY --from=builder --chown=app:app /app /app
WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Run the FastAPI application by default
CMD ["uvicorn" , "--host", "0.0.0.0" , "--port" , "8000" , "src.velsera.webserver.app:app"]
