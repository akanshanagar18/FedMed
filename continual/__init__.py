"""
Continual Federated Learning package for FedMed v2.0.
"""

from continual.replay_buffer import ReplayBuffer
from continual.rehearsal import RehearsalTrainer
from continual.ewc import EWC
from continual.knowledge_distillation import KnowledgeDistillation

__all__ = [
    "ReplayBuffer",
    "RehearsalTrainer",
    "EWC",
    "KnowledgeDistillation",
]
