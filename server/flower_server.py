"""
Module: server.flower_server

Purpose:
Flower server with custom FedAvg strategy that reports aggregation metrics
to the FedMed Dashboard API after each round.

Usage:
    python -m server.flower_server --api-url http://127.0.0.1:8000 --rounds 3
"""

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MetricsReporterStrategy(FedAvg):
    """
    Custom FedAvg strategy that reports round metrics to the Dashboard API.

    After each aggregation round, it POSTs a TrainingMetric payload to
    POST /api/v1/metrics on the monitoring backend.
    """

    def __init__(self, api_url: str = "http://127.0.0.1:8000", experiment_id: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url.rstrip("/")
        self.experiment_id = experiment_id
        self.current_round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client weights using FedAvg, then POST metrics to Dashboard.
        """
        self.current_round = server_round
        start_time = time.time()

        # Perform standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        aggregation_time = time.time() - start_time

        # Collect per-client metrics
        total_loss = 0.0
        total_dice = 0.0
        total_samples = 0
        hospital_ids = []

        for client_proxy, fit_res in results:
            client_metrics = fit_res.metrics
            num_examples = fit_res.num_examples
            total_loss += client_metrics.get("training_loss", 0.0) * num_examples
            total_dice += client_metrics.get("dice_score", 0.0) * num_examples
            total_samples += num_examples
            hospital_id = client_metrics.get("hospital_id", "unknown")
            hospital_ids.append(str(hospital_id))

        avg_loss = total_loss / max(total_samples, 1)
        avg_dice = total_dice / max(total_samples, 1)

        logger.info(
            f"Round {server_round} aggregation complete — "
            f"avg_loss: {avg_loss:.4f}, avg_dice: {avg_dice:.4f}, "
            f"time: {aggregation_time:.2f}s, hospitals: {hospital_ids}"
        )

        # POST metrics to Dashboard API
        self._report_metrics(server_round, avg_loss, avg_dice, hospital_ids)

        return aggregated_parameters, aggregated_metrics

    def _report_metrics(
        self,
        round_number: int,
        avg_loss: float,
        avg_dice: float,
        hospital_ids: List[str],
    ) -> None:
        """Send round metrics to the Dashboard monitoring API."""
        payload = {
            "experiment_id": self.experiment_id,
            "round_number": round_number,
            "training_loss": avg_loss,
            "dice_score": avg_dice,
        }

        try:
            url = f"{self.api_url}/api/v1/metrics"
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info(f"Round {round_number} metrics reported to Dashboard API")
            else:
                logger.warning(
                    f"Dashboard API returned {response.status_code}: {response.text}"
                )
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Could not connect to Dashboard API at {self.api_url}. "
                f"Metrics for round {round_number} will not be persisted."
            )
        except Exception as e:
            logger.error(f"Error reporting metrics: {e}")


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 3,
    min_fit_clients: int = 2,
    min_available_clients: int = 2,
    api_url: str = "http://127.0.0.1:8000",
    experiment_id: str = "default",
):
    """Start the Flower server with the MetricsReporter strategy."""
    strategy = MetricsReporterStrategy(
        api_url=api_url,
        experiment_id=experiment_id,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
    )

    logger.info(
        f"Starting Flower server on {server_address} "
        f"for {num_rounds} rounds (min_clients={min_fit_clients})"
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedMed Flower Server")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum clients per round")
    parser.add_argument("--api-url", type=str, default="http://127.0.0.1:8000", help="Dashboard API URL")
    parser.add_argument("--experiment-id", type=str, default="default", help="Experiment identifier")
    args = parser.parse_args()

    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        api_url=args.api_url,
        experiment_id=args.experiment_id,
    )
