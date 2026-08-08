"""
Module: server.strategies.adapters.flower_adapter

Purpose:
Lightweight, pluggable Flower Adapter wrapping any framework-agnostic BaseStrategy.
Converts Flower protocol primitives (Parameters, ClientProxy, FitRes) to/from native NDArrays and FitResult objects.
Handles real-time metric reporting to the FedMed FastAPI Dashboard API.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import requests

from server.strategies.base import BaseStrategy, EvaluateResult, FitResult
from utils.mlflow_tracker import MLflowTracker
from utils.tensorboard_logger import TensorBoardLogger
from utils.checkpoint_registry import get_checkpoint_registry

logger = logging.getLogger(__name__)


class FlowerStrategyAdapter(fl.server.strategy.Strategy):
    """
    Adapter subclassing `flwr.server.strategy.Strategy` that wraps a pure `BaseStrategy`.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        api_url: str = "http://127.0.0.1:8000",
        experiment_id: str = "default",
    ):
        super().__init__()
        self.strategy = strategy
        self.api_url = api_url.rstrip("/")
        self.experiment_id = experiment_id
        self.current_round = 0
        self.best_dice_score = 0.0
        self.best_round = 0

        # Initialize Milestone J research trackers
        meta = self.strategy.get_metadata()
        self.mlflow_tracker = MLflowTracker(experiment_name="FedMed_v2_Research")
        self.mlflow_tracker.start_run(run_name=f"FL_{meta.name}_{experiment_id}", tags={"strategy": meta.name, "type": "federated"})
        self.mlflow_tracker.log_params({
            "experiment_id": experiment_id,
            "strategy": meta.name,
            "min_fit_clients": getattr(strategy, "min_fit_clients", 2),
            "min_available_clients": getattr(strategy, "min_available_clients", 2),
        })

        self.tb_logger = TensorBoardLogger(experiment_id=experiment_id)
        self.ckpt_registry = get_checkpoint_registry()

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Initialize parameters via underlying strategy or return None."""
        ndarrays = self.strategy.initialize_parameters()
        if ndarrays is not None:
            return ndarrays_to_parameters(ndarrays)
        return None

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters on centralized server dataset (optional)."""
        return None


    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Configure fit instructions for clients via Flower's client manager.
        """
        min_fit = getattr(self.strategy, "min_fit_clients", 2)
        sample_size = max(min_fit, 1)
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_fit)

        config_dict = {}
        if hasattr(self, "latest_metrics") and "encrypted_global_payload" in self.latest_metrics:
            config_dict["encrypted_global_payload"] = self.latest_metrics["encrypted_global_payload"]

        # Build FitIns carrying current global parameters and config
        fit_ins = fl.common.FitIns(parameters, config_dict)
        return [(client, fit_ins) for client in clients]


    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """
        Configure evaluate instructions for clients.
        """
        min_available = getattr(self.strategy, "min_available_clients", 2)
        clients = client_manager.sample(num_clients=min_available, min_num_clients=min_available)
        eval_ins = fl.common.EvaluateIns(parameters, {})
        return [(client, eval_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Translates Flower FitRes into native FitResult list, invokes strategy.aggregate_fit,
        and posts metrics to Dashboard API.
        """
        self.current_round = server_round
        start_time = time.time()

        if not results:
            return None, {}

        # Convert Flower results to native FitResult domain models
        native_results: List[FitResult] = []
        hospital_ids: List[str] = []

        for client_proxy, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            metrics = dict(fit_res.metrics or {})
            native_results.append(
                FitResult(
                    parameters=ndarrays,
                    num_examples=fit_res.num_examples,
                    metrics=metrics,
                    cid=client_proxy.cid,
                )
            )
            h_id = metrics.get("hospital_id", client_proxy.cid)
            hospital_ids.append(str(h_id))

        # Delegate aggregation to framework-agnostic strategy
        aggregated_ndarrays, metrics = self.strategy.aggregate_fit(server_round, native_results, failures)
        self.latest_metrics = metrics
        aggregation_time = time.time() - start_time


        if aggregated_ndarrays is None:
            return None, metrics

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        avg_loss = float(metrics.get("training_loss", 0.0))
        avg_dice = float(metrics.get("dice_score", 0.0))
        iou_score = float(metrics.get("iou_score", 0.0))

        if avg_dice > self.best_dice_score:
            self.best_dice_score = avg_dice
            self.best_round = server_round

        meta = self.strategy.get_metadata()
        logger.info(
            f"[{meta.name}] Round {server_round} aggregation complete — "
            f"avg_loss: {avg_loss:.4f}, avg_dice: {avg_dice:.4f}, "
            f"time: {aggregation_time:.2f}s, hospitals: {hospital_ids}"
        )

        # Milestone J Loggers & Registries
        self.tb_logger.log_round_metrics(
            global_step=server_round,
            training_loss=avg_loss,
            dice=avg_dice,
            iou=iou_score,
            aggregation_time_sec=aggregation_time,
        )
        self.mlflow_tracker.log_metrics({
            "train_loss": avg_loss,
            "dice": avg_dice,
            "iou": iou_score,
            "aggregation_time_sec": aggregation_time,
        }, step=server_round)

        # Checkpoint registration
        try:
            self.ckpt_registry.save_checkpoint(
                model_state_dict={"parameters": aggregated_ndarrays},
                experiment_id=self.experiment_id,
                filename=f"fl_round_{server_round}.pth",
                strategy=meta.name,
                round=server_round,
                dice=avg_dice,
                loss=avg_loss,
            )
        except Exception as e:
            logger.warning(f"Failed to record checkpoint for round {server_round}: {e}")

        # Post metrics to Dashboard API
        self._report_metrics(server_round, avg_loss, avg_dice, hospital_ids)

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Translates Flower EvaluateRes into native EvaluateResult objects and delegates aggregation.
        """
        if not results:
            return None, {}

        native_results: List[EvaluateResult] = []
        for client_proxy, eval_res in results:
            native_results.append(
                EvaluateResult(
                    loss=eval_res.loss,
                    num_examples=eval_res.num_examples,
                    metrics=dict(eval_res.metrics or {}),
                    cid=client_proxy.cid,
                )
            )

        return self.strategy.aggregate_evaluate(server_round, native_results, failures)

    def _report_metrics(
        self,
        round_number: int,
        avg_loss: float,
        avg_dice: float,
        hospital_ids: List[str],
    ) -> None:
        """Send round metrics to Dashboard backend API."""
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
                logger.warning(f"Dashboard API returned {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Could not connect to Dashboard API at {self.api_url}. "
                f"Metrics for round {round_number} will not be persisted."
            )
        except Exception as e:
            logger.error(f"Error reporting metrics: {e}")
