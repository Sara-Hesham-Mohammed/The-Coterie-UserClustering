FROM python:3.12
LABEL authors="Sara"
#ENTRYPOINT ["top", "-b"]
WORKDIR /app


# Install Poetry
RUN pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without virtualenvs, for Docker)
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi


COPY . .

EXPOSE 5000
CMD ["python", "run.py"]
