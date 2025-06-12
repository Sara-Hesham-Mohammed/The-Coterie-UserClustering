# ---------- Base Stage ----------
FROM python:3.12-slim as base

ENV POETRY_VERSION=1.6.1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# ---------- Builder Stage ----------
FROM base as builder

RUN --mount=type=cache,target=/root/.cache \
    pip install "poetry==$POETRY_VERSION"

WORKDIR $PYSETUP_PATH

COPY ./poetry.lock ./pyproject.toml ./

RUN --mount=type=cache,target=$POETRY_HOME/pypoetry/cache \
    poetry install --no-dev

# ---------- Production Stage ----------
FROM base as production

ENV FASTAPI_ENV=production

# Copy installed virtual environment
COPY --from=builder $VENV_PATH $VENV_PATH

# Copy all project source code
COPY ./API /app/API
COPY ./Clusters /app/Clusters
COPY ./Models /app/Models

# Set working directory
WORKDIR /app

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "API.API:app", "--host", "0.0.0.0", "--port", "8000"]
