"""
Module: server.strategies.base

Purpose:
Framework-agnostic Abstract Base Strategy contract and metadata specification for FedMed v2.0.
Decouples federated learning aggregation strategies from specific transport/server frameworks (e.g. Flower).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

# Native parameter representation: list of numpy arrays or torch tensors
NDArray = Any
NDArrays = List[NDArray]


class StrategyMetadata(BaseModel):
    """Metadata specification for production & research federated strategies."""
    name: str = Field(..., description="Canonical name of the strategy (e.g. FedAvg, FedProx)")
    version: str = Field("1.0.0", description="Version of the strategy implementation")
    description: str = Field(..., description="Human-readable summary of the aggregation algorithm")
    supported_features: List[str] = Field(
        default_factory=list,
        description="Features supported (e.g. differential_privacy, homomorphic_encryption, non_iid)"
    )
    supported_config: List[str] = Field(
        default_factory=list,
        description="Configuration parameters accepted by this strategy"
    )
    research_reference: str = Field("", description="Academic paper title, authors, or DOI link")
    default_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default hyperparameter dictionary for this strategy"
    )


class FitResult(BaseModel):
    """Container for client training results passed to aggregate_fit."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parameters: Any  # NDArrays or raw weights
    num_examples: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cid: Optional[str] = None


class EvaluateResult(BaseModel):
    """Container for client evaluation results passed to aggregate_evaluate."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loss: float
    num_examples: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cid: Optional[str] = None



class BaseStrategy(ABC):
    """
    Abstract Base Class for all FedMed Federated Aggregation Strategies.
    
    Implementations must be pure Python/NumPy/PyTorch functions without hard coupling
    to Flower, gRPC, or transport protocol primitives.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def get_metadata(self) -> StrategyMetadata:
        """Returns structured metadata for the strategy."""
        pass

    def initialize_parameters(self) -> Optional[NDArrays]:
        """Optionally return initial global model parameters."""
        try:
            from model.unet3d import UNet3D
            model = UNet3D()
            return [val.cpu().numpy() for _, val in model.state_dict().items()]
        except Exception:
            return None

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[FitResult],
        failures: List[Any],
    ) -> Tuple[Optional[NDArrays], Dict[str, Any]]:
        """
        Aggregate client model parameters after training.
        
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        pass

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[EvaluateResult],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Aggregate client metrics after evaluation.
        
        Returns:
            Tuple of (aggregated_loss, aggregated_metrics)
        """
        pass

    def configure_fit(
        self,
        server_round: int,
        parameters: NDArrays,
        client_manager: Any = None,
    ) -> List[Any]:
        """Configure client fit instructions for the round."""
        return []

    def configure_evaluate(
        self,
        server_round: int,
        parameters: NDArrays,
        client_manager: Any = None,
    ) -> List[Any]:
        """Configure client evaluate instructions for the round."""
        return []

    def get_state(self) -> Dict[str, Any]:
        """Return strategy internal state for checkpointing / resumption."""
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set strategy internal state from checkpoint."""
        pass
