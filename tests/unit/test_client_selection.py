"""
Module: tests.unit.test_client_selection

Purpose:
Unit test suite for Client Selection Engine policies.
"""

import pytest
from server.client_selection import ClientSelectionEngine, ClientProfile


def test_client_selection_policies():
    clients = [
        ClientProfile(cid="c1", num_examples=100, cpu_capacity=2.0, ram_gb=16.0, gpu_available=True, uptime_ratio=0.99, participation_count=5),
        ClientProfile(cid="c2", num_examples=500, cpu_capacity=1.0, ram_gb=8.0, gpu_available=False, uptime_ratio=0.95, participation_count=1),
        ClientProfile(cid="c3", num_examples=300, cpu_capacity=4.0, ram_gb=32.0, gpu_available=True, uptime_ratio=0.98, participation_count=3),
    ]

    # 1. Random Selector
    random_sel = ClientSelectionEngine.get_selector("random")
    s_rand = random_sel.select_clients(clients, num_to_select=2)
    assert len(s_rand) == 2

    # 2. Resource Aware Selector
    res_sel = ClientSelectionEngine.get_selector("resource_aware")
    s_res = res_sel.select_clients(clients, num_to_select=1)
    assert s_res[0].cid == "c3"  # highest cpu (4.0) + ram (32.0) + gpu

    # 3. Data Aware Selector
    data_sel = ClientSelectionEngine.get_selector("data_aware")
    s_data = data_sel.select_clients(clients, num_to_select=1)
    assert s_data[0].cid == "c2"  # highest num_examples (500)

    # 4. Fair Scheduler
    fair_sel = ClientSelectionEngine.get_selector("fair_scheduling")
    s_fair = fair_sel.select_clients(clients, num_to_select=1)
    assert s_fair[0].cid == "c2"  # lowest participation_count (1)
