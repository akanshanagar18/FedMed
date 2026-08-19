"""
Module: server.strategies.adapters.flower_adapter

Purpose:
Lightweight, pluggable Flower Adapter wrapping any framework-agnostic BaseStrategy.
Converts Flower protocol primitives (Parameters, ClientProxy, FitRes) to/from native NDArrays and FitResult objects.
Handles real-time metric reporting to the FedMed FastAPI Dashboard API and SQLite fedmed.db persistence.
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
            "api_url": api_url,
        })
        self.tb_logger = TensorBoardLogger(experiment_id=f"FL_{meta.name}_{experiment_id}", log_dir="runs")

        self.ckpt_registry = get_checkpoint_registry()

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Initializes global model parameters using underlying BaseStrategy."""
        ndarrays = self.strategy.initialize_parameters()
        if ndarrays is not None:
            return ndarrays_to_parameters(ndarrays)
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configures fit instructions for participating Flower clients."""
        self.current_round = server_round
        logger.info(f"--- Strategy Adapter Configured Round {server_round} Fit ---")

        min_fit = getattr(self.strategy, "min_fit_clients", 2)
        if not client_manager.wait_for(num_clients=min_fit, timeout=120):
            logger.warning(
                f"Round {server_round}: Timeout waiting for {min_fit} clients to connect."
            )
            return []

        client_proxies = list(client_manager.all().values())
        cids = [cp.cid for cp in client_proxies]

        selected_proxies = []
        try:
            selected_cids = self.strategy.configure_fit(server_round, cids)
            if isinstance(selected_cids, list) and selected_cids:
                selected_proxies = [cp for cp in client_proxies if cp.cid in selected_cids]
        except Exception:
            selected_proxies = []

        if not selected_proxies:
            min_fit = getattr(self.strategy, "min_fit_clients", 2)
            min_available = getattr(self.strategy, "min_available_clients", min_fit)
            num_available = client_manager.num_available()
            sample_size = max(min_fit, min(num_available, min_available))
            if num_available >= min_fit:
                selected_proxies = client_manager.sample(num_clients=sample_size, min_num_clients=min_fit)
            else:
                selected_proxies = client_proxies

        fit_config = {"server_round": server_round, "strategy": self.strategy.get_metadata().name}
        if hasattr(self, "last_fit_metrics") and isinstance(self.last_fit_metrics, dict):
            for k, v in self.last_fit_metrics.items():
                if isinstance(v, (int, float, str, bytes, bool)):
                    fit_config[k] = v

        fit_ins_list = []
        for cp in selected_proxies:
            fit_ins = fl.common.FitIns(parameters, fit_config)
            fit_ins_list.append((cp, fit_ins))

        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregates local client weights into new global model parameters."""
        start_time = time.time()
        self.current_round = server_round

        if not results:
            logger.warning(f"Round {server_round}: Zero client fit results received.")
            return None, {}

        native_results: List[FitResult] = []
        hospital_ids: List[str] = []
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if fit_res.metrics and "hospital_id" in fit_res.metrics:
                hospital_ids.append(str(fit_res.metrics["hospital_id"]))
            elif cid:
                hospital_ids.append(cid)

            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            native_results.append(
                FitResult(
                    parameters=ndarrays,
                    num_examples=fit_res.num_examples,
                    metrics=dict(fit_res.metrics or {}),
                    cid=cid,
                )
            )

        aggregated_ndarrays, metrics = self.strategy.aggregate_fit(server_round, native_results, failures)
        self.last_fit_metrics = metrics or {}
        if aggregated_ndarrays is None:
            logger.error(f"Round {server_round} strategy aggregation returned None.")
            return None, {}

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        aggregation_time = time.time() - start_time

        avg_loss = float(metrics.get("training_loss", metrics.get("loss", metrics.get("train_loss", 0.0))))
        avg_dice = float(metrics.get("dice_score", metrics.get("dice", 0.0)))
        
        # Exact IoU from metrics or standard mathematical identity IoU = Dice / (2 - Dice)
        if "iou_score" in metrics:
            iou_score = float(metrics["iou_score"])
        elif "iou" in metrics:
            iou_score = float(metrics["iou"])
        else:
            iou_score = float(avg_dice / (2.0 - avg_dice + 1e-8)) if avg_dice > 0 else 0.0

        meta = self.strategy.get_metadata()
        logger.info(
            f"Round {server_round} Aggregation ({meta.name}): Loss={avg_loss:.4f}, Dice={avg_dice:.4f}, "
            f"time: {aggregation_time:.2f}s, hospitals: {hospital_ids}"
        )

        # Milestone J & L Loggers & Registries
        self.tb_logger.log_round_metrics(
            global_step=server_round,
            training_loss=avg_loss,
            dice=avg_dice,
            iou=iou_score,
            aggregation_time_sec=aggregation_time,
        )
        mlflow_metrics = {
            "train_loss": avg_loss,
            "dice": avg_dice,
            "iou": iou_score,
            "aggregation_time_sec": aggregation_time,
        }
        self.mlflow_tracker.log_metrics(mlflow_metrics, step=server_round)

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

        # Post metrics to Dashboard API and persist to fedmed.db
        self._report_metrics(server_round, avg_loss, avg_dice, hospital_ids)

        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
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

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        return []

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluates model parameters on server evaluation dataset if implemented."""
        return None


    def _report_metrics(
        self,
        round_number: int,
        avg_loss: float,
        avg_dice: float,
        hospital_ids: List[str],
    ) -> None:
        """Send round metrics to Dashboard backend API and persist directly to fedmed.db."""
        payload = {
            "experiment_id": self.experiment_id,
            "round_number": round_number,
            "training_loss": avg_loss,
            "dice_score": avg_dice,
        }

        # 1. Post to REST API
        reported_via_api = False
        try:
            url = f"{self.api_url}/api/v1/metrics"
            response = requests.post(url, json=payload, timeout=3)
            if response.status_code in [200, 201]:
                logger.info(f"Round {round_number} metrics reported to Dashboard API")
                reported_via_api = True
        except Exception as e:
            logger.warning(f"Could not report metrics via API ({e}). Falling back to direct DB persistence.")

        # 2. Persist directly to SQLite DB
        try:
            from app.database.session import SessionLocal, init_db
            from app.models.base import TrainingMetricModel
            init_db()
            db = SessionLocal()
            try:
                row = TrainingMetricModel(
                    experiment_id=self.experiment_id,
                    round_number=round_number,
                    training_loss=avg_loss,
                    dice_score=avg_dice,
                )
                db.add(row)
                db.commit()
                logger.info(f"Round {round_number} metrics persisted directly to fedmed.db")
            finally:
                db.close()
        except Exception as dbe:
            logger.error(f"Direct DB persistence failed: {dbe}")
