"""
Module: continual.knowledge_distillation

Purpose:
Teacher-Student Knowledge Distillation for continual federated learning.
Prevents catastrophic forgetting by penalizing divergence between student predictions and previous round teacher logits.
"""

import numpy as np


class KnowledgeDistillation:
    """
    Knowledge Distillation loss calculator.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def compute_distillation_loss(self, student_logits: np.ndarray, teacher_logits: np.ndarray) -> float:
        """
        Computes KL-Divergence distillation loss:
        L_KD = KL( Softmax(teacher / T) || Softmax(student / T) ) * T^2
        """
        # Softmax with temperature
        t_soft = np.exp(teacher_logits / self.temperature)
        t_soft = t_soft / np.sum(t_soft, axis=-1, keepdims=True)

        s_soft = np.exp(student_logits / self.temperature)
        s_soft = s_soft / np.sum(s_soft, axis=-1, keepdims=True)

        kl_div = np.sum(t_soft * (np.log(t_soft + 1e-8) - np.log(s_soft + 1e-8)))
        return float((self.temperature ** 2) * kl_div)
