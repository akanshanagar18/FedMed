# FedMed v2.0 Production Deployment Guide

## 1. Quick Start with Docker Compose

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2.0+

```bash
# 1. Clone Repository & Environment Setup
cp .env.example .env

# 2. Build & Launch Container Suite
docker compose up --build -d

# 3. Check Container Health Status
docker compose ps

# 4. View Container System Logs
docker compose logs -f
```

---

## 2. Operational Control & Container Maintenance

```bash
# Restart single hospital node (Simulate node restart during training)
docker compose restart hospital-beta

# Stop platform cleanly
docker compose down

# Stop platform and remove volumes
docker compose down -v
```

---

## 3. Healthcheck Specifications

- **Backend**: `GET http://localhost:8000/api/v1/health`
- **Flower Server**: TCP port 8080 readiness check
- **Dashboard**: `GET http://localhost:80/`
