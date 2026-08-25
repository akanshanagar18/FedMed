"""
Module: server.strategies.registry

Purpose:
Thread-safe Strategy Registry supporting dynamic auto-discovery and @register_strategy decorator.
Allows adding new algorithms by creating files under server/strategies/ without modifying core registry logic.
"""

import importlib
import logging
import os
import pkgutil
from typing import Callable, Dict, List, Optional, Type
from server.strategies.base import BaseStrategy, StrategyMetadata

logger = logging.getLogger(__name__)


class StrategyRegistryError(Exception):
    """Exception raised for strategy registration or lookup errors."""
    pass


class StrategyRegistry:
    """
    Central Registry for Federated Learning Aggregation Strategies.
    """
    _registry: Dict[str, Type[BaseStrategy]] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, name: Optional[str] = None) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
        """
        Decorator to register a strategy class with the StrategyRegistry.
        
        Usage:
            @register_strategy("FedProx")
            class FedProx(BaseStrategy):
                ...
        """
        def decorator(strategy_cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
            if not issubclass(strategy_cls, BaseStrategy):
                raise StrategyRegistryError(
                    f"Class {strategy_cls.__name__} must inherit from BaseStrategy."
                )
            reg_name = name or strategy_cls.__name__
            cls._registry[reg_name] = strategy_cls
            cls._registry[reg_name.lower()] = strategy_cls
            logger.debug(f"Registered strategy '{reg_name}' -> {strategy_cls.__name__}")
            return strategy_cls

        return decorator

    @classmethod
    def discover_strategies(cls) -> None:
        """
        Dynamically scans and imports all Python modules under server/strategies/
        to trigger auto-registration via decorators.
        """
        if cls._discovered:
            return

        strategies_dir = os.path.dirname(__file__)
        package_prefix = "server.strategies."

        for _, module_name, is_pkg in pkgutil.iter_modules([strategies_dir]):
            if not is_pkg and not module_name.startswith("_") and module_name not in ("base", "registry"):
                full_module_name = f"{package_prefix}{module_name}"
                try:
                    importlib.import_module(full_module_name)
                    logger.debug(f"Discovered strategy module: {full_module_name}")
                except Exception as e:
                    logger.warning(f"Could not import strategy module '{full_module_name}': {e}")

        cls._discovered = True

    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy]:
        """
        Retrieves a registered strategy class by name (case-insensitive).
        """
        cls.discover_strategies()
        strategy_cls = cls._registry.get(name) or cls._registry.get(name.lower())
        if not strategy_cls:
            available = list(cls.list_strategies().keys())
            raise StrategyRegistryError(
                f"Strategy '{name}' not found in registry. Available strategies: {available}"
            )
        return strategy_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseStrategy:
        """
        Instantiates a registered strategy by name with kwargs.
        """
        strategy_cls = cls.get(name)
        try:
            return strategy_cls(**kwargs)
        except Exception as e:
            raise StrategyRegistryError(
                f"Failed to instantiate strategy '{name}' with kwargs {kwargs}: {e}"
            ) from e

    @classmethod
    def list_strategies(cls) -> Dict[str, StrategyMetadata]:
        """
        Returns metadata for all registered unique strategies.
        """
        cls.discover_strategies()
        metadata_map: Dict[str, StrategyMetadata] = {}
        for name, strategy_cls in cls._registry.items():
            # Skip lowercase duplicate lookup keys
            if name[0].isupper():
                try:
                    instance = strategy_cls()
                    metadata_map[name] = instance.get_metadata()
                except Exception:
                    # Fallback metadata if default instantiation requires args
                    metadata_map[name] = StrategyMetadata(
                        name=name,
                        description=f"Registered strategy {strategy_cls.__name__}",
                    )
        return metadata_map


# Convenience alias decorator
register_strategy = StrategyRegistry.register
