"""
Module: server.strategies.fedavg

Purpose:
Pure, framework-agnostic implementation of the Federated Averaging (FedAvg) algorithm.
Supports both standard plaintext weighted parameter averaging and TenSEAL CKKS Homomorphic Encrypted ciphertext aggregation.
Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from privacy.aggregation import aggregate_encrypted_updates
from privacy.communication import deserialize_encrypted_payload, serialize_encrypted_payload
from privacy.context import create_ckks_context, get_public_context
from privacy.decrypt import decrypt_model_parameters
from server.strategies.base import BaseStrategy, EvaluateResult, FitResult, NDArrays, StrategyMetadata
from server.strategies.registry import register_strategy

logger = logging.getLogger(__name__)


@register_strategy("FedAvg")
class FedAvg(BaseStrategy):
    """
    Standard Federated Averaging (FedAvg) strategy.
    Performs sample-weighted elementwise averaging of client parameter updates
    or homomorphic ciphertext vector aggregation.
    """

    def __init__(
        self,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.he_context = None

    def get_metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="FedAvg",
            version="1.0.0",
            description="Federated Averaging (McMahan et al., 2017) baseline aggregation strategy.",
            supported_features=["weighted_averaging", "homomorphic_encryption"],
            supported_config=["min_fit_clients", "min_available_clients"],
            research_reference="https://arxiv.org/abs/1602.05629",
            default_parameters={
                "min_fit_clients": 2,
                "min_available_clients": 2,
            },
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[NDArrays], Dict[str, Any]]:
        """
        Computes sample-weighted elementwise average of parameters or performs ciphertext aggregation.
        """
        if not results:
            return None, {}

        # Calculate total sample count
        total_samples = sum(res.num_examples for res in results)
        if total_samples == 0:
            return None, {}

        # Check if incoming updates contain encrypted payloads
        has_encryption = any(
            isinstance(res.metrics, dict) and "encrypted_payload" in res.metrics
            for res in results
        )

        if has_encryption:
            logger.info(f"[FedAvg] Homomorphic Encryption payload detected across {len(results)} client updates. Performing ciphertext aggregation...")
            encrypted_results = []
            shapes = None
            total_enc_time_ms = 0.0
            total_ciphertext_bytes = 0

            for res in results:
                payload_str = res.metrics["encrypted_payload"]
                payload_dict = deserialize_encrypted_payload(payload_str)
                encrypted_results.append((payload_dict["encrypted_chunks"], res.num_examples))

                if shapes is None:
                    shapes = payload_dict["shapes"]

                total_enc_time_ms += payload_dict.get("encryption_time_ms", 0.0)
                total_ciphertext_bytes += payload_dict.get("ciphertext_size_bytes", 0)

            # Lazy initialize Public Evaluation CKKS Context (no secret key)
            if self.he_context is None:
                private_ctx = create_ckks_context(poly_modulus_degree=8192)
                self.he_context = get_public_context(private_ctx)
                assert self.he_context.is_private() is False, "Server context must be strictly public!"

            agg_he_res = aggregate_encrypted_updates(self.he_context, encrypted_results, shapes)
            serialized_agg_payload = serialize_encrypted_payload(agg_he_res)

            # Return dummy zero arrays for parameters so zero plaintext weights cross wire
            aggregated_params = [np.zeros(shape, dtype=np.float32) for shape in shapes]

            # Metrics carrying encrypted global ciphertext payload to clients
            total_loss = sum(res.metrics.get("training_loss", 0.0) * res.num_examples for res in results)
            total_dice = sum(res.metrics.get("dice_score", 0.0) * res.num_examples for res in results)

            metrics = {
                "training_loss": total_loss / total_samples,
                "dice_score": total_dice / total_samples,
                "num_clients": len(results),
                "total_samples": total_samples,
                "he_enabled": True,
                "encrypted_global_payload": serialized_agg_payload,
                "encryption_time_ms": total_enc_time_ms / len(results),
                "aggregation_time_ms": float(agg_he_res["aggregation_time_ms"]),
                "ciphertext_size_bytes": int(agg_he_res["ciphertext_size_bytes"]),
            }
            return aggregated_params, metrics


        else:
            # Standard Plaintext Aggregation
            first_params = results[0].parameters
            if first_params is None:
                return None, {}

            weighted_weights: List[np.ndarray] = [
                np.zeros_like(layer, dtype=np.float64) for layer in first_params
            ]

            for res in results:
                weight = res.num_examples / total_samples
                for i, layer in enumerate(res.parameters):
                    weighted_weights[i] += layer * weight

            aggregated_params: NDArrays = [
                layer.astype(first_params[i].dtype) if isinstance(first_params[i], np.ndarray) else layer
                for i, layer in enumerate(weighted_weights)
            ]

            total_loss = sum(res.metrics.get("training_loss", 0.0) * res.num_examples for res in results)
            total_dice = sum(res.metrics.get("dice_score", 0.0) * res.num_examples for res in results)

            metrics = {
                "training_loss": total_loss / total_samples,
                "dice_score": total_dice / total_samples,
                "num_clients": len(results),
                "total_samples": total_samples,
                "he_enabled": False,
            }
            return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[EvaluateResult],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Computes sample-weighted average loss and dice score for validation.
        """
        if not results:
            return None, {}

        total_samples = sum(res.num_examples for res in results)
        if total_samples == 0:
            return None, {}

        weighted_loss = sum(res.loss * res.num_examples for res in results) / total_samples
        weighted_dice = sum(res.metrics.get("dice_score", 0.0) * res.num_examples for res in results) / total_samples

        metrics = {
            "loss": weighted_loss,
            "dice_score": weighted_dice,
            "total_samples": total_samples,
        }
        return weighted_loss, metrics
