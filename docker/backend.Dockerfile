# FedMed v2.0 FastAPI Backend Production Dockerfile
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
COPY --chown=fedmeduser:fedmeduser dashboard/backend /app/dashboard/backend
COPY --chown=fedmeduser:fedmeduser configs /app/configs
COPY --chown=fedmeduser:fedmeduser privacy /app/privacy
COPY --chown=fedmeduser:fedmeduser data /app/data
COPY --chown=fedmeduser:fedmeduser model /app/model
COPY --chown=fedmeduser:fedmeduser server /app/server
COPY --chown=fedmeduser:fedmeduser client /app/client

ENV PATH=/home/fedmeduser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER fedmeduser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "dashboard/backend", "--host", "0.0.0.0", "--port", "8000"]
