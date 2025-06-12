FROM python:3.12-slim

ENV POETRY_VERSION=1.6.1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no virtualenvs, no root package)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy app source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "API.API:app", "--host", "0.0.0.0", "--port", "8000"]
