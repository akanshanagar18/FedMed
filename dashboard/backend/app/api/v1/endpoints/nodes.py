"""
Module: dashboard.backend.app.api.v1.endpoints.nodes

Purpose:
REST API endpoints for Hospital Node Resilience, Status Monitoring, and Heartbeats.
Exposes GET /api/v1/nodes and POST /api/v1/nodes/heartbeat per FedMed v2.0 specification.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory store for real-time node resilience tracking
_NODE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "hospital_alpha": {
        "hospital_id": "hospital_alpha",
        "status": "ONLINE",
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "active_round": 0,
        "reconnect_count": 0,
        "training_state": "idle",
    },
    "hospital_beta": {
        "hospital_id": "hospital_beta",
        "status": "ONLINE",
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "active_round": 0,
        "reconnect_count": 0,
        "training_state": "idle",
    },
    "hospital_gamma": {
        "hospital_id": "hospital_gamma",
        "status": "ONLINE",
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "active_round": 0,
        "reconnect_count": 0,
        "training_state": "idle",
    },
}


class NodeHeartbeatRequest(BaseModel):
    hospital_id: str
    status: str = Field("ONLINE", description="Node status: ONLINE, OFFLINE, RECONNECTING, FAILED, ACTIVE")
    active_round: int = 0
    reconnect_count: int = 0
    training_state: str = "idle"


@router.get("", response_model=Dict[str, Any])
def get_hospital_nodes() -> Dict[str, Any]:
    """
    Returns live list of hospital nodes, statuses, last_seen timestamps, and resilience metrics.
    """
    nodes_list = list(_NODE_REGISTRY.values())
    return {
        "status": "success",
        "total_nodes": len(nodes_list),
        "online_nodes": sum(1 for n in nodes_list if n["status"] in ["ONLINE", "ACTIVE"]),
        "nodes": nodes_list,
    }


@router.post("/heartbeat", response_model=Dict[str, Any])
def record_node_heartbeat(heartbeat: NodeHeartbeatRequest) -> Dict[str, Any]:
    """
    Receives node status update/heartbeat from hospital client nodes or orchestrator.
    """
    h_id = heartbeat.hospital_id
    if h_id not in _NODE_REGISTRY:
        _NODE_REGISTRY[h_id] = {
            "hospital_id": h_id,
            "status": heartbeat.status.upper(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "active_round": heartbeat.active_round,
            "reconnect_count": heartbeat.reconnect_count,
            "training_state": heartbeat.training_state,
        }
    else:
        node = _NODE_REGISTRY[h_id]
        node["status"] = heartbeat.status.upper()
        node["last_seen"] = datetime.now(timezone.utc).isoformat()
        node["active_round"] = heartbeat.active_round
        node["reconnect_count"] = heartbeat.reconnect_count
        node["training_state"] = heartbeat.training_state

    return {
        "status": "success",
        "message": f"Heartbeat recorded for node '{h_id}'",
        "node": _NODE_REGISTRY[h_id],
    }
