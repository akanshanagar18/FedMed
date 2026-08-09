# FedMed v2.0 — Makefile Entrypoint Standardizer

.PHONY: help backend frontend demo simulation dev stop clean test test-unit test-integration test-e2e benchmark mlflow tensorboard docker compose compose-down k8s-deploy

PYTHON := ./venv/bin/python3
PYTEST := ./venv/bin/pytest
UVICORN := ./venv/bin/uvicorn
MLFLOW := ./venv/bin/mlflow
TENSORBOARD := ./venv/bin/tensorboard

help:
	@echo "======================================================================"
	@echo "🏥 FEDMED v2.0 — ENTERPRISE FEDERATED AI PLATFORM ENTRYPOINTS"
	@echo "======================================================================"
	@echo "  make demo              - Execute 1-command automated enterprise demo"
	@echo "  make backend           - Launch production FastAPI backend (port 8000)"
	@echo "  make frontend          - Launch React frontend dev server (port 3000)"
	@echo "  make dev               - Launch backend in auto-reload development mode"
	@echo "  make simulation        - Run multi-hospital Flower FL simulation"
	@echo "  make test              - Run complete pytest suite (unit, integration, e2e)"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests only"
	@echo "  make test-e2e          - Run E2E smoke tests only"
	@echo "  make benchmark         - Run multi-experiment benchmark matrix"
	@echo "  make mlflow            - Start MLflow Tracking Server UI (port 5000)"
	@echo "  make tensorboard       - Start TensorBoard log visualizer (port 6006)"
	@echo "  make docker            - Build backend production Docker image"
	@echo "  make compose           - Launch containerized platform stack via Docker Compose"
	@echo "  make compose-down      - Shutdown Docker Compose platform stack"
	@echo "  make k8s-deploy        - Deploy platform to Kubernetes cluster"
	@echo "  make stop              - Stop running backend/server processes"
	@echo "  make clean             - Remove pycache, temporary DBs, and build artifacts"
	@echo "======================================================================"

backend:
	PYTHONPATH=dashboard/backend:. $(UVICORN) app.main:app --app-dir dashboard/backend --host 0.0.0.0 --port 8000

dev:
	PYTHONPATH=dashboard/backend:. $(UVICORN) app.main:app --app-dir dashboard/backend --host 0.0.0.0 --port 8000 --reload

frontend:
	cd dashboard/frontend && npm run dev

demo:
	PYTHONPATH=. $(PYTHON) demo.py

simulation:
	PYTHONPATH=. $(PYTHON) scripts/run_simulation.py

test:
	$(PYTEST)

test-unit:
	$(PYTEST) tests/unit

test-integration:
	$(PYTEST) tests/integration

test-e2e:
	$(PYTEST) tests/e2e

benchmark:
	PYTHONPATH=. $(PYTHON) scripts/run_benchmarks.py --quick

mlflow:
	$(MLFLOW) ui --host 0.0.0.0 --port 5000

tensorboard:
	$(TENSORBOARD) --logdir runs --port 6006

docker:
	docker build -f docker/backend.Dockerfile -t fedmed-backend:latest .

compose:
	docker compose up --build -d

compose-down:
	docker compose down -v

k8s-deploy:
	kubectl apply -f k8s/fedmed-namespace.yaml
	kubectl apply -f k8s/

stop:
	@pkill -f uvicorn || true
	@pkill -f mlflow || true
	@pkill -f tensorboard || true
	@echo "Stopped server processes."

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name "*.db-journal" -delete
	rm -rf .pytest_cache
	@echo "Cleaned transient build files and caches."
