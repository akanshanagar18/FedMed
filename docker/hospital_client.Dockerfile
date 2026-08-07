# FedMed v2.0 Hospital Client Node Production Dockerfile
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
COPY --chown=fedmeduser:fedmeduser client /app/client
COPY --chown=fedmeduser:fedmeduser configs /app/configs
COPY --chown=fedmeduser:fedmeduser data /app/data
COPY --chown=fedmeduser:fedmeduser model /app/model
COPY --chown=fedmeduser:fedmeduser privacy /app/privacy

ENV PATH=/home/fedmeduser/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER fedmeduser

HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os, sys; sys.exit(0 if os.path.exists('/app/client/flower_client.py') else 1)" || exit 1

CMD ["python", "-m", "client.flower_client"]
