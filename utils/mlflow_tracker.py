"""
Module: utils.mlflow_tracker

Purpose:
Production MLflow Integration Engine for FedMed v2.0.
Tracks federated and centralized experiment runs, hyperparameters, segmentation research metrics,
and uploads model checkpoints, plots, benchmark reports, and logs to MLflow.
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional

from utils.reproducibility import get_git_metadata, get_dependency_versions

logger = logging.getLogger("mlflow_tracker")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow library not found. MLflow logging will be disabled or mocked in fallback mode.")


class MLflowTracker:
    """
    Manages lifecycle of MLflow runs for FedMed v2.0 baseline training, FL simulations, and benchmark matrix sweeps.
    """

    def __init__(
        self,
        experiment_name: str = "FedMed_v2_Research",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
        self.artifact_location = artifact_location
        self.current_run = None
        self.is_active = MLFLOW_AVAILABLE

        if self.is_active:
            try:
                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"Initialized MLflow Tracker (URI='{self.tracking_uri}', Experiment='{self.experiment_name}')")
            except Exception as e:
                logger.warning(f"Failed to set MLflow experiment/URI: {e}. Disabling MLflow tracking.")
                self.is_active = False

    def start_run(
        self,
        run_name: str,
        tags: Optional[Dict[str, Any]] = None,
        nested: bool = False
    ) -> Optional[Any]:
        """Starts a new MLflow tracking run."""
        if not self.is_active:
            return None

        try:
            full_tags = get_git_metadata()
            if tags:
                full_tags.update({str(k): str(v) for k, v in tags.items()})

            if mlflow.active_run() and not nested:
                try:
                    mlflow.end_run()
                except Exception:
                    pass
            self.current_run = mlflow.start_run(run_name=run_name, tags=full_tags, nested=nested)
            logger.info(f"Started MLflow Run '{run_name}' (ID: {self.current_run.info.run_id})")
            return self.current_run
        except Exception as e:
            logger.warning(f"Error starting MLflow run: {e}")
            return None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Logs experiment hyperparameters and system configuration."""
        if not self.is_active or not self.current_run:
            return

        formatted_params = {}
        for k, v in params.items():
            formatted_params[str(k)] = str(v) if v is not None else "N/A"

        try:
            mlflow.log_params(formatted_params)
            logger.debug(f"Logged {len(formatted_params)} parameters to MLflow.")
        except Exception as e:
            logger.warning(f"Error logging parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs metrics (Dice, IoU, Loss, Hausdorff, Precision, Recall, etc.) for a specific step or round."""
        if not self.is_active or not self.current_run:
            return

        valid_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not (v != v):  # Check for NaN
                valid_metrics[str(k)] = float(v)

        try:
            mlflow.log_metrics(valid_metrics, step=step)
            logger.debug(f"Logged {len(valid_metrics)} metrics to MLflow at step {step}.")
        except Exception as e:
            logger.warning(f"Error logging metrics to MLflow: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Uploads a single local file (e.g. best_model.pth, report.pdf) to MLflow artifact store."""
        if not self.is_active or not self.current_run:
            return

        try:
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
                logger.debug(f"Logged artifact '{local_path}' to MLflow.")
        except Exception as e:
            logger.warning(f"Error logging artifact to MLflow: {e}")

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Uploads an entire local directory of artifacts to MLflow artifact store."""
        if not self.is_active or not self.current_run:
            return

        if not os.path.exists(local_dir):
            logger.warning(f"Cannot upload non-existent artifact directory: '{local_dir}'")
            return

        try:
            mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
            logger.info(f"Uploaded artifact directory '{local_dir}' to MLflow.")
        except Exception as e:
            logger.warning(f"Error uploading artifact directory '{local_dir}' to MLflow: {e}")

    def log_governance_event(self, event_name: str, details: Dict[str, Any]) -> None:
        """Logs governance event details and stage transitions as MLflow tags & params."""
        if not self.is_active or not self.current_run:
            return
        try:
            for k, v in details.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"gov_{k}", float(v))
                else:
                    mlflow.set_tag(f"gov_{k}", str(v))
            logger.info(f"Logged governance event '{event_name}' to MLflow.")
        except Exception as e:
            logger.warning(f"Error logging governance event to MLflow: {e}")

    def log_drift_metrics(self, node_id: str, drift_metrics: Dict[str, float]) -> None:
        """Logs distribution drift metrics (MMD, KS, Wasserstein, PSI) per node to MLflow."""
        if not self.is_active or not self.current_run:
            return
        try:
            for k, v in drift_metrics.items():
                mlflow.log_metric(f"drift_{node_id}_{k}", float(v))
            logger.info(f"Logged drift metrics for node '{node_id}' to MLflow.")
        except Exception as e:
            logger.warning(f"Error logging drift metrics to MLflow: {e}")

    def end_run(self, status: str = "FINISHED") -> None:

        """Ends the currently active MLflow run."""
        if not self.is_active or not self.current_run:
            return

        try:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow Run (Status: {status}).")
            self.current_run = None
        except Exception as e:
            logger.warning(f"Error ending MLflow run: {e}")


def get_default_mlflow_tracker() -> MLflowTracker:
    """Factory helper to obtain default MLflowTracker instance."""
    return MLflowTracker()
