"""
Module: dashboard.backend.app.services.hospital

Purpose:
Defines the interface for the HospitalService.
Abstracts logic for registering and querying hospital nodes.

TODO:
- [ ] Implement actual logic.
"""

from app.schemas.node import HospitalStatus
from typing import List


class HospitalService:
    """Interface for managing Hospital statuses."""
    
    @classmethod
    async def get_all_hospitals(cls) -> List[HospitalStatus]:
        """Returns the status of all registered hospitals."""
        return []
