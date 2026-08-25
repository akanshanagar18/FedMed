"""
Module: tests.unit.test_network_simulator

Purpose:
Unit test suite for Network Simulator module.
"""

import pytest
from simulation.network_simulator import NetworkSimulator, NetworkProfile


def test_network_simulator_lan_profile():
    profile = NetworkProfile(name="LAN", latency_ms=0.5, bandwidth_mbps=1000.0)
    sim = NetworkSimulator(profile)

    delay = sim.calculate_transmission_delay(payload_size_bytes=1000000)
    assert delay > 0.0
    metrics = sim.get_metrics()
    assert metrics["network_profile"] == "LAN"
    assert metrics["packets_sent"] == 1


def test_network_simulator_from_yaml_preset():
    sim = NetworkSimulator.from_yaml("configs/network/mobile_4g.yaml")
    metrics = sim.get_metrics()
    assert metrics["network_profile"] == "Mobile_4G"
    assert metrics["latency_ms"] == 50.0
