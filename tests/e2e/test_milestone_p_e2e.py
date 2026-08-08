"""
Module: tests.e2e.test_milestone_p_e2e

Purpose:
End-to-End simulation test for Milestone P Large-Scale Distributed Federated Systems,
Asynchronous FL, Communication Compression, and Network Simulation.
"""

import pytest
import numpy as np

from server.strategies.fedasync import FedAsync
from communication.quantization import UniformQuantizer
from communication.sparsification import Sparsifier
from communication.sign_sgd import SignSGDCompressor
from server.client_selection import ClientSelectionEngine, ClientProfile
from simulation.network_simulator import NetworkSimulator


def test_milestone_p_e2e_distributed_systems_pipeline():
    # 1. Client selection
    clients = [ClientProfile(cid=f"h_{i}", num_examples=100 + i * 10, cpu_capacity=2.0) for i in range(25)]
    selector = ClientSelectionEngine.get_selector("resource_aware")
    selected_clients = selector.select_clients(clients, num_to_select=10)
    assert len(selected_clients) == 10

    # 2. Network simulation
    sim = NetworkSimulator.from_yaml("configs/network/mobile_4g.yaml")
    weights = [np.random.randn(10, 10).astype(np.float32)]
    payload_size = sum(w.nbytes for w in weights)
    delay = sim.calculate_transmission_delay(payload_size)
    assert delay > 0.0

    # 3. Compression engine
    quantizer = UniformQuantizer(bits=8)
    compressed_payload = quantizer.compress(weights)
    assert compressed_payload.compression_ratio >= 1.0

    decompressed_weights = quantizer.decompress(compressed_payload)
    assert decompressed_weights[0].shape == weights[0].shape

    # 4. Asynchronous FL Strategy
    strategy = FedAsync(alpha=0.5, staleness_func="polynomial")
    assert strategy.get_metadata().name == "FedAsync"
