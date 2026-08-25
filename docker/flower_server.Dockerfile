# FedMed v2.0 Central Flower Server Production Dockerfile
FROM python:3.9-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim AS runner

WORKDIR /app

RUN useradd -m -u 1000 fedmeduser

COPY --from=builder /root/.local /home/fedmeduser/.local
COPY --chown=fedmeduser:fedmeduser . /app

ENV PATH=/home/fedmeduser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER fedmeduser

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.connect(('127.0.0.1', 8080))" || exit 1

CMD ["python", "-m", "server.flower_server", "--address", "0.0.0.0:8080"]
