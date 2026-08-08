"""
Foundation Model Adapters package for FedMed v2.0.
"""

from foundation.sam_adapter import SAMAdapter
from foundation.medsam_adapter import MedSAMAdapter
from foundation.dinov2_adapter import DINOv2Adapter

__all__ = [
    "SAMAdapter",
    "MedSAMAdapter",
    "DINOv2Adapter",
]
