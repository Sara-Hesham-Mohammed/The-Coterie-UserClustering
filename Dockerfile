FROM python:3.12
LABEL authors="Sara"
#ENTRYPOINT ["top", "-b"]
WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
