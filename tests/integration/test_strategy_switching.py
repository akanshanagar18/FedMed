"""
Integration tests for strategy switching via configuration and REST API exposure.
"""

from fastapi.testclient import TestClient
import pytest
from configs.loader import load_config
from dashboard.backend.app.main import app
from server.strategies import FlowerStrategyAdapter, StrategyRegistry


@pytest.fixture
def api_client():
    return TestClient(app)


def test_strategy_switching_via_config_fedavg():
    """Verify loading fedavg.yaml resolves FedAvg strategy via StrategyRegistry."""
    app_cfg = load_config("configs/fedavg.yaml")
    strat_name = app_cfg.federated.get_strategy_name()
    strat_params = app_cfg.federated.get_strategy_parameters()

    assert strat_name == "FedAvg"
    strategy_inst = StrategyRegistry.create(strat_name, **strat_params)
    assert strategy_inst.get_metadata().name == "FedAvg"

    adapter = FlowerStrategyAdapter(strategy=strategy_inst)
    assert adapter.strategy.get_metadata().name == "FedAvg"


def test_strategy_switching_via_config_fedprox():
    """Verify loading fedprox.yaml resolves FedProx strategy with proximal_mu via StrategyRegistry."""
    app_cfg = load_config("configs/fedprox.yaml")
    strat_name = app_cfg.federated.get_strategy_name()
    strat_params = app_cfg.federated.get_strategy_parameters()

    assert strat_name == "FedProx"
    assert app_cfg.federated.proximal_mu == 0.01

    strategy_inst = StrategyRegistry.create(strat_name, **strat_params)
    assert strategy_inst.get_metadata().name == "FedProx"
    assert getattr(strategy_inst, "proximal_mu") == 0.01

    adapter = FlowerStrategyAdapter(strategy=strategy_inst)
    assert adapter.strategy.get_metadata().name == "FedProx"


def test_dashboard_strategies_api_endpoints(api_client):
    """Verify /api/v1/strategies REST API returns registered strategy metadata."""
    response = api_client.get("/api/v1/strategies")
    assert response.status_code == 200
    json_data = response.json()
    assert "retrieved successfully" in json_data["message"]
    strategies = json_data["data"]["strategies"]

    assert "FedAvg" in strategies
    assert "FedProx" in strategies

    # Detailed strategy metadata check
    fedprox_resp = api_client.get("/api/v1/strategies/FedProx")
    assert fedprox_resp.status_code == 200
    prox_data = fedprox_resp.json()["data"]
    assert prox_data["name"] == "FedProx"
    assert "proximal_regularization" in prox_data["supported_features"]
