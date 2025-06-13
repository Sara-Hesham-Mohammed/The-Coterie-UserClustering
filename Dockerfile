FROM python:3.12-slim


# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source code
COPY . .

# Expose port and set start command
EXPOSE 8000
CMD ["uvicorn", "API.API:app", "--host", "0.0.0.0", "--port", "8000"]
