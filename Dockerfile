FROM python:3.12-slim
LABEL authors="Sara"

WORKDIR /app

# Installing system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    pip install --no-cache-dir poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml poetry.lock* ./

# Disabling virtualenv creation (for Docker) and installing dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

COPY . .

EXPOSE 8000

# Run FastAPI app via uvicorn
CMD ["uvicorn", "API.API:app", "--host", "0.0.0.0", "--port", "8000"]
