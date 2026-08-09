"""
Module: configs.policy_engine

Purpose:
Enterprise Policy Engine for FedMed v2.0.
Eliminates hardcoded magic numbers by reading dynamic policy configurations from YAML files
(policy.yaml, governance.yaml, deployment.yaml, sla.yaml, privacy.yaml, strategy.yaml, health.yaml).
Supports live policy reloads.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("policy_engine")


class EnterprisePolicyEngine:
    """
    Central Dynamic Policy Engine loading policy parameters from YAML manifests.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnterprisePolicyEngine, cls).__new__(cls)
            cls._instance.policy_config_path = os.path.abspath("configs/policy.yaml")
            cls._instance.policies: Dict[str, Any] = {}
            cls._instance.reload_count: int = 0
            cls._instance.reload_policies()
        return cls._instance

    def reload_policies(self) -> Dict[str, Any]:
        """Reads policy YAML configuration from disk and updates policy memory cache."""
        if os.path.exists(self.policy_config_path):
            try:
                with open(self.policy_config_path, "r", encoding="utf-8") as f:
                    self.policies = yaml.safe_load(f) or {}
                self.reload_count += 1
                logger.info(f"Loaded policy configuration from '{self.policy_config_path}' (Reload #{self.reload_count})")
            except Exception as e:
                logger.error(f"Error parsing policy configuration '{self.policy_config_path}': {e}")
                self.policies = self._get_fallback_defaults()
        else:
            logger.warning(f"Policy YAML path '{self.policy_config_path}' not found. Using fallback defaults.")
            self.policies = self._get_fallback_defaults()

        return self.policies

    def _get_fallback_defaults(self) -> Dict[str, Any]:
        """Provides default fallback policy dictionary if file is unreadable."""
        return {
            "governance": {
                "significance_level": 0.05,
                "psi_threshold": 0.20,
                "mmd_drift_warning_threshold": 0.10,
                "mmd_drift_critical_threshold": 0.25,
                "min_dice_canary_improvement": 0.005,
            },
            "deployment": {
                "default_strategy": "CANARY",
                "canary_initial_traffic_pct": 10.0,
                "auto_rollback_on_error_spike": True,
            },
            "sla": {
                "max_epsilon": 10.0,
                "max_delta": 0.0001,
                "min_participation_rate": 0.75,
                "max_latency_ms": 2000.0,
            },
            "privacy": {
                "default_scheme": "TenSEAL_CKKS",
                "dp_enabled": True,
            },
            "strategy": {
                "default_fl_strategy": "FedAvg",
                "fallback_strategy": "FedProx",
            },
            "health": {
                "heartbeat_interval_sec": 10,
                "max_retry_attempts": 3,
            },
        }

    def get_policy(self, section: str, key: str, default: Any = None) -> Any:
        """Retrieves a specific policy parameter value."""
        sec = self.policies.get(section, {})
        return sec.get(key, default)

    def get_governance_policy(self) -> Dict[str, Any]:

        return self.policies.get("governance", {})

    def get_sla_policy(self) -> Dict[str, Any]:
        return self.policies.get("sla", {})

    def list_rules(self) -> List[Any]:
        """Returns list of active policy rules."""
        rules = []
        class _PolicyRule:
            def __init__(self, sec, k, v):
                self.section = sec
                self.key = k
                self.value = v
            def to_dict(self):
                return {"section": self.section, "key": self.key, "value": self.value}

        for sec, items in self.policies.items():
            if isinstance(items, dict):
                for k, v in items.items():
                    rules.append(_PolicyRule(sec, k, v))
        return rules


global_policy_engine = EnterprisePolicyEngine()

