"""
Module: continual.rehearsal

Purpose:
Interleaved rehearsal trainer combining new task batches with historical replay samples.
"""

from typing import Any, Dict, List
import numpy as np
from continual.replay_buffer import ReplayBuffer


class RehearsalTrainer:
    """
    Interleaved Rehearsal Manager.
    """

    def __init__(self, replay_buffer: ReplayBuffer, replay_ratio: float = 0.2):
        self.replay_buffer = replay_buffer
        self.replay_ratio = float(replay_ratio)

    def prepare_interleaved_batch(self, current_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.replay_buffer.size() == 0:
            return current_batch

        num_replay = int(len(current_batch) * self.replay_ratio)
        replay_samples = self.replay_buffer.sample_batch(batch_size=num_replay)
        return current_batch + replay_samples
