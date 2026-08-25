"""
Unit tests for node resilience state tracking logic.
"""

import pytest
from app.api.v1.endpoints.nodes import _NODE_REGISTRY, NodeHeartbeatRequest, record_node_heartbeat, get_hospital_nodes


def test_node_heartbeat_recording():
    """Verify node heartbeat recording and status retrieval."""
    hb = NodeHeartbeatRequest(hospital_id="test_hospital", status="ACTIVE", active_round=1, reconnect_count=0)
    res = record_node_heartbeat(hb)
    assert res["status"] == "success"
    assert res["node"]["status"] == "ACTIVE"

    nodes_data = get_hospital_nodes()
    assert nodes_data["status"] == "success"
    assert any(n["hospital_id"] == "test_hospital" for n in nodes_data["nodes"])
