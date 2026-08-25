"""
Module: dashboard.backend.app.api.v1.endpoints.strategies

Purpose:
REST API endpoints exposing Strategy Engine metadata, registered strategies, descriptions,
and hyperparameter specifications to the FedMed Dashboard frontend.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from common.schemas import SuccessResponse
from server.strategies import StrategyRegistry, StrategyRegistryError

router = APIRouter()


@router.get("", response_model=SuccessResponse)
async def list_registered_strategies():
    """Returns metadata summary for all registered strategy plugins in StrategyRegistry."""
    strategies_metadata = StrategyRegistry.list_strategies()
    # Convert Pydantic metadata objects to dicts
    serialized_data = {
        name: meta.model_dump(mode="json")
        for name, meta in strategies_metadata.items()
    }
    return SuccessResponse(
        message="Registered federated strategies retrieved successfully",
        data={
            "count": len(serialized_data),
            "strategies": serialized_data,
        },
    )


@router.get("/{strategy_name}", response_model=SuccessResponse)
async def get_strategy_details(strategy_name: str):
    """Returns detailed metadata and default hyperparameters for a specific strategy."""
    try:
        strategy_cls = StrategyRegistry.get(strategy_name)
        instance = strategy_cls()
        metadata = instance.get_metadata()
        return SuccessResponse(
            message=f"Strategy '{strategy_name}' metadata retrieved successfully",
            data=metadata.model_dump(mode="json"),
        )
    except StrategyRegistryError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inspecting strategy '{strategy_name}': {e}")
