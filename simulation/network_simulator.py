"""
Module: simulation.network_simulator

Purpose:
Network profile simulator for federated learning systems.
Simulates bandwidth throttling, network latency, jitter, packet loss, and client node disconnection probabilities.
"""

import time
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from utils.logger import get_logger

logger = get_logger("network_simulator")


class NetworkProfile:
    """Network configuration parameters."""

    def __init__(
        self,
        name: str = "LAN",
        latency_ms: float = 0.5,
        bandwidth_mbps: float = 1000.0,
        packet_loss_pct: float = 0.0,
        jitter_ms: float = 0.1,
        drop_probability: float = 0.0,
    ):
        self.name = name
        self.latency_ms = float(latency_ms)
        self.bandwidth_mbps = float(bandwidth_mbps)
        self.packet_loss_pct = float(packet_loss_pct)
        self.jitter_ms = float(jitter_ms)
        self.drop_probability = float(drop_probability)


class NetworkSimulator:
    """
    Network Simulator computing transmission delays and packet drop events.
    """

    def __init__(self, profile: Optional[NetworkProfile] = None):
        self.profile = profile or NetworkProfile()
        self.total_bytes_transferred: int = 0
        self.total_packets_sent: int = 0
        self.total_packets_dropped: int = 0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "NetworkSimulator":
        p = Path(yaml_path)
        if not p.exists():
            logger.warning(f"Network config YAML '{yaml_path}' not found. Using LAN default.")
            return cls(NetworkProfile())

        with open(p, "r") as f:
            data = yaml.safe_load(f).get("network", {})

        profile = NetworkProfile(
            name=data.get("profile", "Custom"),
            latency_ms=data.get("latency_ms", 0.5),
            bandwidth_mbps=data.get("bandwidth_mbps", 1000.0),
            packet_loss_pct=data.get("packet_loss_pct", 0.0),
            jitter_ms=data.get("jitter_ms", 0.1),
            drop_probability=data.get("drop_probability", 0.0),
        )
        return cls(profile)

    def calculate_transmission_delay(self, payload_size_bytes: int) -> float:
        """
        Computes total estimated transmission delay in seconds:
        delay = latency + jitter + (size / bandwidth)
        """
        self.total_bytes_transferred += payload_size_bytes
        self.total_packets_sent += 1

        # Check for simulated drop event
        if np.random.rand() < self.profile.drop_probability:
            self.total_packets_dropped += 1

        base_latency_sec = self.profile.latency_ms / 1000.0
        jitter_sec = max(0.0, np.random.normal(0, self.profile.jitter_ms / 1000.0))

        # Bandwidth delay: bytes / (Mbps * 10^6 / 8)
        bandwidth_bytes_per_sec = (self.profile.bandwidth_mbps * 1e6) / 8.0
        transfer_delay_sec = payload_size_bytes / max(1.0, bandwidth_bytes_per_sec)

        return float(base_latency_sec + jitter_sec + transfer_delay_sec)

    def simulate_delay(self, payload_size_bytes: int, scale_factor: float = 0.001) -> float:
        """Optionally sleeps or returns simulated delay."""
        delay = self.calculate_transmission_delay(payload_size_bytes)
        # Scaled delay for fast unit/simulation execution
        time.sleep(min(delay * scale_factor, 0.05))
        return delay

    def get_metrics(self) -> Dict[str, Any]:
        drop_rate = float(self.total_packets_dropped / self.total_packets_sent) if self.total_packets_sent > 0 else 0.0
        return {
            "network_profile": self.profile.name,
            "latency_ms": self.profile.latency_ms,
            "bandwidth_mbps": self.profile.bandwidth_mbps,
            "total_megabytes_transferred": float(self.total_bytes_transferred / (1024 * 1024)),
            "packets_sent": self.total_packets_sent,
            "packets_dropped": self.total_packets_dropped,
            "packet_drop_rate": drop_rate,
        }
