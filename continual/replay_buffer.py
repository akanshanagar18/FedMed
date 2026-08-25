"""
Module: continual.replay_buffer

Purpose:
Prioritized experience replay buffer for continual federated learning.
Stores representative historical patient samples to prevent catastrophic forgetting.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer maintaining historical task samples.
    """

    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []

    def add_sample(self, sample_id: str, image_data: np.ndarray, label_mask: np.ndarray, priority: float = 1.0) -> None:
        if len(self.buffer) >= self.capacity:
            # Remove lowest priority sample
            self.buffer.sort(key=lambda s: s["priority"])
            self.buffer.pop(0)

        self.buffer.append({
            "sample_id": sample_id,
            "image": image_data,
            "mask": label_mask,
            "priority": priority,
        })

    def sample_batch(self, batch_size: int = 16) -> List[Dict[str, Any]]:
        if not self.buffer:
            return []
        size = min(len(self.buffer), batch_size)
        indices = np.random.choice(len(self.buffer), size=size, replace=False)
        return [self.buffer[i] for i in indices]

    def size(self) -> int:
        return len(self.buffer)
