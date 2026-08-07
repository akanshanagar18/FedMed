# FedMed Strategy Engine Architecture Specification

## Overview
The FedMed Strategy Engine provides a production-grade, framework-agnostic plugin architecture for federated learning aggregation algorithms.
It decouples mathematical parameter aggregation, control variates, and state management from transport protocols and server frameworks (such as Flower).

---

## 1. Architecture Diagram
```mermaid
graph TD
    Client[Hospital Client Node] -->|FL Updates| FlowerAdapter[FlowerStrategyAdapter]
    FlowerAdapter -->|Native Parameters| PureStrategy[BaseStrategy Implementation]
    
    subgraph Core Engine
        BaseStrategy[server.strategies.base.BaseStrategy]
        Registry[server.strategies.registry.StrategyRegistry]
        FedAvg[server.strategies.fedavg.FedAvg]
        FedProx[server.strategies.fedprox.FedProx]
    end

    PureStrategy --> BaseStrategy
    FedAvg --> BaseStrategy
    FedProx --> BaseStrategy
    
    Registry -->|Creates| FedAvg
    Registry -->|Creates| FedProx
    
    FlowerAdapter -->|Round Metrics| DashboardAPI[FastAPI Dashboard REST API]
```

---

## 2. Class Hierarchy
```mermaid
classDiagram
    class BaseStrategy {
        <<abstract>>
        +initialize_parameters() NDArrays
        +aggregate_fit(server_round, results, failures) Tuple~NDArrays, dict~
        +aggregate_evaluate(server_round, results, failures) Tuple~float, dict~
        +configure_fit(server_round, parameters, client_manager) List
        +configure_evaluate(server_round, parameters, client_manager) List
        +get_metadata() StrategyMetadata
        +get_state() dict
        +set_state(state) void
    }

    class FedAvg {
        +aggregate_fit(...)
        +aggregate_evaluate(...)
        +get_metadata() StrategyMetadata
    }

    class FedProx {
        -float proximal_mu
        +aggregate_fit(...)
        +aggregate_evaluate(...)
        +get_metadata() StrategyMetadata
    }

    class StrategyRegistry {
        -dict _registry
        +register(name, cls)
        +get(name) Type~BaseStrategy~
        +create(name, **kwargs) BaseStrategy
        +list_strategies() dict
        +discover_strategies()
    }

    class FlowerStrategyAdapter {
        -BaseStrategy strategy
        -str api_url
        -str experiment_id
        +aggregate_fit(...)
        +aggregate_evaluate(...)
        +_report_metrics(...)
    }

    BaseStrategy <|-- FedAvg
    BaseStrategy <|-- FedProx
    StrategyRegistry o-- BaseStrategy
    FlowerStrategyAdapter o-- BaseStrategy
```

---

## 3. Dependency Graph
```mermaid
graph LR
    YAML[YAML Configuration] --> ConfigLoader[configs.loader]
    ConfigLoader --> StrategyRegistry[server.strategies.registry]
    StrategyRegistry --> Strategies[server.strategies.*]
    Strategies --> BaseStrategy[server.strategies.base]
    BaseStrategy --> Adapter[server.strategies.adapters.FlowerStrategyAdapter]
    Adapter --> Server[server.flower_server]
    Adapter --> API[Dashboard REST API /api/v1/strategies]
```

---

## 4. Migration Strategy
1. **Backward Compatibility**: Existing configuration files referencing `strategy: FedAvg` or `strategy: FedProx` at top-level or under `federated:` work without modification.
2. **Auto-Discovery**: Any new strategy file placed in `server/strategies/new_algorithm.py` with `@register_strategy("NewAlgorithm")` will be auto-discovered by `StrategyRegistry` with zero core code edits.
3. **Structured Configuration**: Advanced strategy configs can specify nested parameters:
```yaml
federated:
  strategy:
    name: FedProx
    parameters:
      proximal_mu: 0.05
```

---

## 5. Verification Commands
```bash
# Unit & Integration Tests
PYTHONPATH=. venv/bin/pytest tests/unit/test_strategy_registry.py tests/unit/test_fedavg.py tests/unit/test_fedprox.py tests/integration/test_strategy_switching.py

# Execute FedAvg Simulation
python scripts/run_simulation.py --config configs/fedavg.yaml

# Execute FedProx Simulation
python scripts/run_simulation.py --config configs/fedprox.yaml
```

---

## 6. Dashboard REST API Endpoints
- `GET /api/v1/strategies` -> Lists metadata summaries for all registered strategy plugins.
- `GET /api/v1/strategies/{strategy_name}` -> Returns detailed metadata, academic references, and hyperparameter specifications for the selected algorithm.
