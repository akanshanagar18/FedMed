"""
Module: dashboard.backend.app.services.hospital

Purpose:
HospitalService handles registration and status tracking of hospital nodes.
"""

from sqlalchemy.orm import Session
from app.models.base import HospitalNodeModel
from app.schemas.node import HospitalStatus
from typing import List
from datetime import datetime


class HospitalService:
    """CRUD operations for Hospital nodes."""

    @classmethod
    def register_or_update(cls, db: Session, status: HospitalStatus) -> HospitalNodeModel:
        """Register a new hospital or update an existing one."""
        existing = (
            db.query(HospitalNodeModel)
            .filter(HospitalNodeModel.hospital_id == status.hospital_id)
            .first()
        )
        if existing:
            existing.name = status.name
            existing.connection_status = status.connection_status
            existing.client_latency_ms = status.client_latency_ms
            existing.last_seen = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing
        else:
            row = HospitalNodeModel(
                hospital_id=status.hospital_id,
                name=status.name,
                connection_status=status.connection_status,
                client_latency_ms=status.client_latency_ms,
                last_seen=datetime.utcnow(),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return row

    @classmethod
    def get_all_hospitals(cls, db: Session) -> List[HospitalStatus]:
        """Returns the status of all registered hospitals."""
        rows = db.query(HospitalNodeModel).all()
        return [
            HospitalStatus(
                hospital_id=r.hospital_id,
                name=r.name,
                connection_status=r.connection_status,
                client_latency_ms=r.client_latency_ms,
                last_seen=r.last_seen,
            )
            for r in rows
        ]
