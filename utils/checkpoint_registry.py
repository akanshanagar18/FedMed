"""
Module: utils.checkpoint_registry

Purpose:
Production Checkpoint Registry Module for FedMed v2.0.
Registers, indexes, hashes, and manages PyTorch model state checkpoints along with rich JSON sidecar metadata.
Supports queries for latest model, best performing model by metric, resuming training state, and listing registry entries.
"""

import os
import hashlib
import json
import time
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("checkpoint_registry")


class CheckpointMetadata:
    """Dataclass wrapper representing metadata for a saved checkpoint entry."""

    def __init__(
        self,
        checkpoint_id: str,
        file_path: str,
        experiment_id: str,
        benchmark_id: Optional[str] = None,
        strategy: str = "FedAvg",
        epoch: Optional[int] = None,
        round: Optional[int] = None,
        dice: float = 0.0,
        loss: float = 0.0,
        privacy_settings: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        file_hash: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.checkpoint_id = checkpoint_id
        self.file_path = file_path
        self.experiment_id = experiment_id
        self.benchmark_id = benchmark_id or "standalone"
        self.strategy = strategy
        self.epoch = epoch
        self.round = round
        self.dice = dice
        self.loss = loss
        self.privacy_settings = privacy_settings or {"dp_enabled": False, "he_enabled": False}
        self.config = config or {}
        self.file_hash = file_hash or ""
        self.timestamp = timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "file_path": self.file_path,
            "experiment_id": self.experiment_id,
            "benchmark_id": self.benchmark_id,
            "strategy": self.strategy,
            "epoch": self.epoch,
            "round": self.round,
            "dice": self.dice,
            "loss": self.loss,
            "privacy_settings": self.privacy_settings,
            "config": self.config,
            "file_hash": self.file_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(
            checkpoint_id=data.get("checkpoint_id", ""),
            file_path=data.get("file_path", ""),
            experiment_id=data.get("experiment_id", "default"),
            benchmark_id=data.get("benchmark_id"),
            strategy=data.get("strategy", "FedAvg"),
            epoch=data.get("epoch"),
            round=data.get("round"),
            dice=data.get("dice", 0.0),
            loss=data.get("loss", 0.0),
            privacy_settings=data.get("privacy_settings"),
            config=data.get("config"),
            file_hash=data.get("file_hash"),
            timestamp=data.get("timestamp"),
        )


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class CheckpointRegistry:
    """
    Central Checkpoint Registry managing PyTorch model weights and registry metadata index in `checkpoints/`.
    """

    def __init__(self, registry_dir: str = "checkpoints"):
        self.registry_dir = os.path.abspath(registry_dir)
        os.makedirs(self.registry_dir, exist_ok=True)
        self.index_file = os.path.join(self.registry_dir, "registry.json")
        self._entries: List[CheckpointMetadata] = []
        self._load_index()

    def _load_index(self) -> None:
        """Loads registry entries from JSON index file."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._entries = [CheckpointMetadata.from_dict(item) for item in data]
            except Exception as e:
                logger.warning(f"Failed to read checkpoint index '{self.index_file}': {e}")
                self._entries = []
        else:
            self._entries = []

    def _save_index(self) -> None:
        """Persists registry entries to JSON index file."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._entries], f, indent=2)
        except Exception as e:
            logger.error(f"Error persisting checkpoint index to '{self.index_file}': {e}")

    def save_checkpoint(
        self,
        model_state_dict: Dict[str, Any],
        experiment_id: str,
        filename: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        strategy: str = "FedAvg",
        epoch: Optional[int] = None,
        round: Optional[int] = None,
        dice: float = 0.0,
        loss: float = 0.0,
        privacy_settings: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
    ) -> CheckpointMetadata:
        """
        Saves a PyTorch state dict and registers metadata entry with computed SHA-256 hash.
        """
        ts_suffix = int(time.time())
        chk_id = f"chk_{experiment_id}_{round or epoch or 'latest'}_{ts_suffix}"
        fn = filename or f"{chk_id}.pth"
        full_path = os.path.join(self.registry_dir, fn)

        checkpoint_payload = {
            "model_state_dict": model_state_dict,
            "experiment_id": experiment_id,
            "epoch": epoch,
            "round": round,
            "dice": dice,
            "loss": loss,
        }
        if optimizer_state_dict is not None:
            checkpoint_payload["optimizer_state_dict"] = optimizer_state_dict

        torch.save(checkpoint_payload, full_path)
        file_hash = compute_sha256(full_path)

        meta = CheckpointMetadata(
            checkpoint_id=chk_id,
            file_path=full_path,
            experiment_id=experiment_id,
            benchmark_id=benchmark_id,
            strategy=strategy,
            epoch=epoch,
            round=round,
            dice=dice,
            loss=loss,
            privacy_settings=privacy_settings,
            config=config,
            file_hash=file_hash,
        )

        # Save individual JSON sidecar file alongside .pth
        sidecar_path = full_path.rsplit(".", 1)[0] + ".json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)

        # Remove previous entry with same file_path if exists
        self._entries = [e for e in self._entries if e.file_path != full_path]
        self._entries.append(meta)
        self._save_index()

        logger.info(f"Registered Checkpoint '{chk_id}' at '{full_path}' (Dice={dice:.4f}, Hash={file_hash[:10]}...)")
        return meta

    def list_checkpoints(
        self,
        experiment_id: Optional[str] = None,
        benchmark_id: Optional[str] = None,
    ) -> List[CheckpointMetadata]:
        """Lists all registered checkpoints, optionally filtered by experiment or benchmark ID."""
        results = self._entries
        if experiment_id:
            results = [e for e in results if e.experiment_id == experiment_id]
        if benchmark_id:
            results = [e for e in results if e.benchmark_id == benchmark_id]
        return results

    def get_latest_checkpoint(self, experiment_id: Optional[str] = None) -> Optional[CheckpointMetadata]:
        """Returns the most recently created checkpoint."""
        entries = self.list_checkpoints(experiment_id=experiment_id)
        if not entries:
            return None
        return sorted(entries, key=lambda x: (x.timestamp, x.round or 0, x.epoch or 0), reverse=True)[0]

    def get_best_checkpoint(
        self,
        experiment_id: Optional[str] = None,
        metric: str = "dice",
    ) -> Optional[CheckpointMetadata]:
        """Returns the checkpoint with highest dice score (or lowest loss)."""
        entries = self.list_checkpoints(experiment_id=experiment_id)
        if not entries:
            return None
        if metric == "loss":
            return sorted(entries, key=lambda x: x.loss)[0]
        return sorted(entries, key=lambda x: x.dice, reverse=True)[0]

    def load_checkpoint_for_resume(self, checkpoint_path_or_id: str) -> Tuple[Dict[str, Any], CheckpointMetadata]:
        """
        Loads PyTorch checkpoint payload and matching metadata for training resumption.
        """
        target_path = checkpoint_path_or_id
        matching_meta = None

        for e in self._entries:
            if e.checkpoint_id == checkpoint_path_or_id or e.file_path == checkpoint_path_or_id:
                target_path = e.file_path
                matching_meta = e
                break

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Checkpoint file '{target_path}' not found on disk.")

        payload = torch.load(target_path, map_location="cpu")
        if matching_meta is None:
            sidecar = target_path.rsplit(".", 1)[0] + ".json"
            if os.path.exists(sidecar):
                with open(sidecar, "r", encoding="utf-8") as f:
                    matching_meta = CheckpointMetadata.from_dict(json.load(f))
            else:
                matching_meta = CheckpointMetadata(
                    checkpoint_id=os.path.basename(target_path),
                    file_path=target_path,
                    experiment_id="resume",
                )

        logger.info(f"Loaded checkpoint for resume from '{target_path}'")
        return payload, matching_meta


_global_checkpoint_registry: Optional[CheckpointRegistry] = None


def get_checkpoint_registry(registry_dir: str = "checkpoints") -> CheckpointRegistry:
    """Returns singleton CheckpointRegistry instance."""
    global _global_checkpoint_registry
    if _global_checkpoint_registry is None:
        _global_checkpoint_registry = CheckpointRegistry(registry_dir=registry_dir)
    return _global_checkpoint_registry
