FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data \
    && mkdir -p /app/logs \
    && mkdir -p /app/plots

VOLUME ["/app/data", "/app/logs", "/app/plots"]

CMD ["python", "src/server.py"]