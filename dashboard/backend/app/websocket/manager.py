"""
Module: dashboard.backend.app.websocket.manager

Purpose:
Provides the WebSocket ConnectionManager infrastructure.
Handles active client connections and broadcasts telemetry events 
without implementing domain-specific business logic.

TODO:
- [ ] Implement Redis Pub/Sub backend if scaling beyond a single worker.
"""

from fastapi import WebSocket
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for live telemetry."""
    
    def __init__(self):
        # In a real distributed setup, replace this with Redis Pub/Sub.
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, event_name: str, payload: Dict[str, Any]):
        """Broadcasts a structured JSON event to all connected clients."""
        message = json.dumps({"event": event_name, "data": payload})
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
