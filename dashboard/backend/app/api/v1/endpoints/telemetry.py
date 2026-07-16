"""
Module: dashboard.backend.app.api.v1.endpoints.telemetry

Purpose:
WebSocket endpoint for real-time telemetry streaming to the React frontend.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websocket.manager import manager
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws")
async def telemetry_endpoint(websocket: WebSocket):
    """
    Establishes a WebSocket connection for streaming live metrics and node status.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection open, wait for client messages if any
            data = await websocket.receive_text()
            logger.info(f"Received message from client: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
