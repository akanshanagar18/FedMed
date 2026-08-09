.PHONY: demo docker-up k8s-deploy test test-unit benchmark

demo:
	./venv/bin/python3 demo.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

k8s-deploy:
	kubectl apply -f k8s/fedmed-namespace.yaml
	kubectl apply -f k8s/

test:
	./venv/bin/pytest

test-unit:
	./venv/bin/pytest tests/unit

benchmark:
	./venv/bin/python3 -c "from evaluation.benchmark_runner import global_benchmark_runner; print(global_benchmark_runner.run_benchmark_matrix()['markdown_report'])"
