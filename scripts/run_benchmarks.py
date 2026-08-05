"""
Module: scripts.run_benchmarks

Purpose:
Production-Grade Automated Benchmark Framework for FedMed v2.0.
Supports loading matrix parameters from YAML via --config flag, executing FL runs sequentially,
persisting metadata, and generating publication-grade result artifacts under results/benchmark_<id>/.

Usage:
  python scripts/run_benchmarks.py --config configs/benchmark.yaml --quick
"""

import argparse
import csv
import dataclasses
import itertools
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Tuple

from configs.loader import load_config, AppConfig, ConfigValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [BENCHMARK] - %(message)s")
logger = logging.getLogger("benchmark")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


@dataclasses.dataclass
class MatrixConfig:
    strategies: List[str]
    partitions: List[str]
    seeds: List[int]
    num_rounds: int = 3
    num_clients: int = 2


def generate_experiment_matrix(config: MatrixConfig, benchmark_id: str) -> List[Dict[str, Any]]:
    """Generate combinations of matrix sweeps."""
    matrix = []
    index = 1
    for strategy, partition, seed in itertools.product(config.strategies, config.partitions, config.seeds):
        exp_id = f"{benchmark_id}_exp_{index:03d}_{strategy}_{seed}"
        matrix.append({
            "experiment_id": exp_id,
            "name": f"{strategy} | {partition} | Seed {seed}",
            "strategy_name": strategy,
            "partition_strategy": partition,
            "seed": seed,
            "num_rounds": config.num_rounds,
            "num_clients": config.num_clients,
        })
        index += 1
    return matrix


def register_benchmark_backend(api_url: str, benchmark_id: str, name: str, total_experiments: int) -> None:
    """Register benchmark suite with FastAPI backend."""
    payload = {
        "benchmark_id": benchmark_id,
        "name": name,
        "description": "Automated hyperparameter matrix sweep",
        "status": "running",
        "total_experiments": total_experiments,
    }
    try:
        url = f"{api_url.rstrip('/')}/api/v1/benchmarks"
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info(f"Registered benchmark '{benchmark_id}' in backend")
    except Exception as e:
        logger.warning(f"Could not register benchmark in backend: {e}")


def execute_single_experiment(exp_cfg: Dict[str, Any], api_url: str) -> Dict[str, Any]:
    """
    Executes a single experiment run by orchestrating simulation pipeline
    and collecting final metrics.
    """
    exp_id = exp_cfg["experiment_id"]
    logger.info(f"Executing Experiment [{exp_id}] ({exp_cfg['name']})...")
    start_time = time.time()

    # Launch run_simulation.py as subprocess
    sim_script = os.path.join(PROJECT_ROOT, "scripts", "run_simulation.py")
    proc = subprocess.Popen(
        [sys.executable, sim_script],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=60)
        exit_code = proc.returncode
        runtime = time.time() - start_time

        if exit_code == 0:
            status_str = "completed"
        else:
            status_str = "failed"
            logger.error(f"Experiment [{exp_id}] failed: {stderr}")

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        status_str = "failed"
        runtime = time.time() - start_time

    # Fetch metric results from backend
    best_dice = 0.0
    avg_loss = 0.0
    conv_round = 0
    try:
        metrics_url = f"{api_url.rstrip('/')}/api/v1/metrics/default"
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                metrics = body.get("data", [])
                if metrics:
                    dices = [m.get("dice_score", 0.0) for m in metrics if m.get("dice_score") is not None]
                    losses = [m.get("training_loss", 0.0) for m in metrics if m.get("training_loss") is not None]
                    if dices:
                        best_dice = max(dices)
                        conv_round = dices.index(best_dice) + 1
                    if losses:
                        avg_loss = sum(losses) / len(losses)
    except Exception:
        pass

    return {
        "experiment_id": exp_id,
        "strategy_name": exp_cfg["strategy_name"],
        "partition_strategy": exp_cfg["partition_strategy"],
        "seed": exp_cfg["seed"],
        "status": status_str,
        "best_dice": round(best_dice, 4),
        "avg_loss": round(avg_loss, 4),
        "convergence_round": conv_round,
        "runtime_sec": round(runtime, 2),
    }


def generate_benchmark_artifacts(
    benchmark_id: str, name: str, results: List[Dict[str, Any]], output_dir: str
) -> None:
    """Generates summary.json, metrics.csv, leaderboard.csv, and comparison.md."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Leaderboard CSV
    leaderboard_csv = os.path.join(output_dir, "leaderboard.csv")
    results_sorted = sorted(results, key=lambda x: x["best_dice"], reverse=True)
    
    with open(leaderboard_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Experiment ID", "Strategy", "Partition", "Seed", "Best Dice", "Avg Loss", "Convergence Round", "Runtime (s)", "Status"])
        for rank, r in enumerate(results_sorted, start=1):
            writer.writerow([
                rank, r["experiment_id"], r["strategy_name"], r["partition_strategy"],
                r["seed"], r["best_dice"], r["avg_loss"], r["convergence_round"], r["runtime_sec"], r["status"]
            ])

    # 2. Metrics CSV
    metrics_csv = os.path.join(output_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment ID", "Strategy", "Partition", "Best Dice", "Avg Loss", "Runtime (s)"])
        for r in results:
            writer.writerow([r["experiment_id"], r["strategy_name"], r["partition_strategy"], r["best_dice"], r["avg_loss"], r["runtime_sec"]])

    # 3. Summary JSON
    summary_json = os.path.join(output_dir, "summary.json")
    best_run = results_sorted[0] if results_sorted else {}
    summary_data = {
        "benchmark_id": benchmark_id,
        "name": name,
        "timestamp": datetime.utcnow().isoformat(),
        "total_experiments": len(results),
        "completed_experiments": sum(1 for r in results if r["status"] == "completed"),
        "best_experiment_id": best_run.get("experiment_id"),
        "best_dice_score": best_run.get("best_dice"),
        "average_dice_score": round(sum(r["best_dice"] for r in results) / max(len(results), 1), 4),
    }
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)

    # 4. Comparison Markdown Report
    comparison_md = os.path.join(output_dir, "comparison.md")
    with open(comparison_md, "w") as f:
        f.write(f"# Benchmark Results Report: {name}\n\n")
        f.write(f"**Benchmark ID:** `{benchmark_id}`  \n")
        f.write(f"**Total Experiments:** {len(results)}  \n")
        f.write(f"**Best Performing Strategy:** `{best_run.get('strategy_name')}` (Dice: `{best_run.get('best_dice')}`)  \n\n")
        f.write("## Leaderboard Table\n\n")
        f.write("| Rank | Strategy | Partition | Seed | Best Dice | Avg Loss | Runtime (s) |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for rank, r in enumerate(results_sorted, start=1):
            f.write(f"| {rank} | `{r['strategy_name']}` | `{r['partition_strategy']}` | {r['seed']} | **{r['best_dice']}** | {r['avg_loss']} | {r['runtime_sec']}s |\n")

    logger.info(f"Artifacts successfully generated in '{output_dir}':")
    logger.info(f"  - {leaderboard_csv}")
    logger.info(f"  - {metrics_csv}")
    logger.info(f"  - {summary_json}")
    logger.info(f"  - {comparison_md}")


def main():
    parser = argparse.ArgumentParser(description="FedMed v2.0 Benchmark Framework")
    parser.add_argument("--config", type=str, default=None, help="Path to modular YAML config file")
    parser.add_argument("--name", type=str, default="FedMed Strategy Benchmark", help="Benchmark suite name")
    parser.add_argument("--api-url", type=str, default=None, help="Backend API URL override")
    parser.add_argument("--quick", action="store_true", help="Run a quick 2-experiment matrix sweep for testing")
    args = parser.parse_args()

    # Load YAML config
    try:
        app_cfg: AppConfig = load_config(args.config)
        logger.info(f"Loaded configuration cleanly for Benchmark Engine: {args.config or 'configs/default.yaml'}")
    except ConfigValidationError as e:
        logger.error(f"CRITICAL: Configuration error:\n{e}")
        sys.exit(1)

    api_url = args.api_url or app_cfg.server.api_url
    benchmark_id = f"bm_{int(time.time())}"
    output_dir = os.path.join(RESULTS_DIR, f"benchmark_{benchmark_id}")

    if args.quick:
        matrix_cfg = MatrixConfig(
            strategies=["FedAvg", "FedProx"],
            partitions=["IID"],
            seeds=[42],
            num_rounds=app_cfg.federated.num_rounds,
            num_clients=app_cfg.federated.min_clients,
        )
    else:
        matrix_cfg = MatrixConfig(
            strategies=app_cfg.benchmark.strategies,
            partitions=app_cfg.benchmark.partitions,
            seeds=app_cfg.benchmark.seeds,
            num_rounds=app_cfg.federated.num_rounds,
            num_clients=app_cfg.federated.min_clients,
        )

    matrix = generate_experiment_matrix(matrix_cfg, benchmark_id)
    logger.info("==========================================================")
    logger.info(f" Launching FedMed Benchmark Suite [{benchmark_id}]")
    logger.info(f" Matrix Size: {len(matrix)} Experiments")
    logger.info("==========================================================")

    register_benchmark_backend(api_url, benchmark_id, args.name, len(matrix))

    results = []
    for exp_cfg in matrix:
        res = execute_single_experiment(exp_cfg, api_url)
        results.append(res)

    generate_benchmark_artifacts(benchmark_id, args.name, results, output_dir)
    logger.info("==========================================================")
    logger.info(f" SUCCESS: Benchmark [{benchmark_id}] completed!")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()
