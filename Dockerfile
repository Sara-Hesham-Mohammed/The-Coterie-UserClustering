FROM python:3.10
LABEL authors="Sara"
#ENTRYPOINT ["top", "-b"]
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
