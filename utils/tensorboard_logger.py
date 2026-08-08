"""
Module: utils.tensorboard_logger

Purpose:
Production TensorBoard SummaryWriter Logging Engine for FedMed v2.0.
Logs metrics, training loss, validation loss, MONAI segmentation scores, privacy budgets,
gradient norms, memory consumption, and timing benchmarks directly to logdir 'runs/'.
"""

import os
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger("tensorboard_logger")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        logger.warning("TensorBoard SummaryWriter not available. TensorBoard logging will operate in dummy mode.")


class DummyWriter:
    """Fallback no-op writer when TensorBoard library is not present."""
    def add_scalar(self, *args, **kwargs): pass
    def add_scalars(self, *args, **kwargs): pass
    def add_histogram(self, *args, **kwargs): pass
    def add_hparams(self, *args, **kwargs): pass
    def close(self): pass
    def flush(self): pass


class TensorBoardLogger:
    """
    Manages TensorBoard SummaryWriter for FedMed experiments, logging under 'runs/<experiment_id>'.
    """

    def __init__(self, experiment_id: str, log_dir: str = "runs"):
        self.experiment_id = experiment_id
        self.log_path = os.path.join(log_dir, experiment_id)
        os.makedirs(self.log_path, exist_ok=True)

        if TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=self.log_path)
                logger.info(f"Initialized TensorBoard SummaryWriter at '{self.log_path}'")
            except Exception as e:
                logger.warning(f"Could not instantiate SummaryWriter: {e}. Using DummyWriter.")
                self.writer = DummyWriter()
        else:
            self.writer = DummyWriter()

    def log_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """Logs a single scalar metric tag at global_step."""
        try:
            self.writer.add_scalar(tag=tag, scalar_value=float(scalar_value), global_step=global_step)
        except Exception as e:
            logger.warning(f"Error writing scalar tag '{tag}' to TensorBoard: {e}")

    def log_round_metrics(
        self,
        global_step: int,
        training_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        dice: Optional[float] = None,
        iou: Optional[float] = None,
        learning_rate: Optional[float] = None,
        privacy_budget: Optional[float] = None,
        grad_norm: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        round_time_sec: Optional[float] = None,
        aggregation_time_sec: Optional[float] = None,
        encryption_time_sec: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Logs standard FedMed research metrics required by Milestone J PART B.
        """
        metrics_mapping = {
            "Loss/Train": training_loss,
            "Loss/Validation": val_loss,
            "Metrics/Dice": dice,
            "Metrics/IoU": iou,
            "Hyperparameters/LearningRate": learning_rate,
            "Privacy/Epsilon": privacy_budget,
            "Optimization/GradientNorm": grad_norm,
            "Hardware/GPUMemoryMB": gpu_memory_mb,
            "Performance/RoundTimeSec": round_time_sec,
            "Performance/AggregationTimeSec": aggregation_time_sec,
            "Performance/EncryptionTimeSec": encryption_time_sec,
        }

        if extra_metrics:
            for k, v in extra_metrics.items():
                metrics_mapping[f"Custom/{k}"] = v

        for tag, val in metrics_mapping.items():
            if val is not None and isinstance(val, (int, float)) and not (val != val):
                self.log_scalar(tag, val, global_step)

    def flush(self) -> None:
        try:
            self.writer.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.writer.flush()
            self.writer.close()
            logger.info(f"Closed TensorBoard writer for experiment '{self.experiment_id}'")
        except Exception:
            pass
