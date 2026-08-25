"""
Module: dashboard.backend.app.websocket.manager

Purpose:
Provides the WebSocket ConnectionManager infrastructure.
Handles active client connections and broadcasts telemetry events 
without implementing domain-specific business logic.
"""

from fastapi import WebSocket
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for live telemetry."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")
        try:
            await websocket.send_text(json.dumps({
                "event": "connected",
                "status": "ONLINE",
                "message": "FedMed WebSocket Telemetry Active",
            }))
        except Exception as e:
            logger.warning(f"Failed to send welcome message: {e}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, event_name_or_payload: Any, payload: Optional[Dict[str, Any]] = None):
        """Broadcasts a structured JSON event to all connected clients."""
        if payload is not None:
            message = json.dumps({"event": str(event_name_or_payload), "data": payload})
        elif isinstance(event_name_or_payload, dict):
            message = json.dumps(event_name_or_payload)
        else:
            message = json.dumps({"event": "telemetry", "data": event_name_or_payload})

        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(connection)
                
        # Clean up dead connections
        for client in disconnected_clients:
            self.disconnect(client)


# Singleton instance
manager = ConnectionManager()
