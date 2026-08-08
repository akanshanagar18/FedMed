"""
Module: tests.unit.test_continual_learning

Purpose:
Unit test suite for continual learning engine (ReplayBuffer, RehearsalTrainer, EWC, KnowledgeDistillation).
"""

import numpy as np
import pytest

from continual.replay_buffer import ReplayBuffer
from continual.rehearsal import RehearsalTrainer
from continual.ewc import EWC
from continual.knowledge_distillation import KnowledgeDistillation


def test_replay_buffer_and_rehearsal():
    buf = ReplayBuffer(capacity=10)
    buf.add_sample("s1", np.zeros((1, 128, 128)), np.zeros((1, 128, 128)))
    buf.add_sample("s2", np.ones((1, 128, 128)), np.ones((1, 128, 128)))

    assert buf.size() == 2
    batch = buf.sample_batch(batch_size=1)
    assert len(batch) == 1

    trainer = RehearsalTrainer(buf, replay_ratio=0.5)
    current_batch = [{"sample_id": "c1"}]
    interleaved = trainer.prepare_interleaved_batch(current_batch)
    assert len(interleaved) >= 1


def test_ewc_penalty_calculation():
    ewc = EWC(ewc_lambda=400.0)
    params = [np.array([1.0, 2.0], dtype=np.float32)]
    grads = [np.array([0.1, 0.2], dtype=np.float32)]

    ewc.update_fisher_matrix(params, grads)
    assert ewc.optimal_params is not None

    current_params = [np.array([1.1, 2.1], dtype=np.float32)]
    penalty = ewc.compute_penalty(current_params)
    assert penalty > 0.0


def test_knowledge_distillation():
    kd = KnowledgeDistillation(temperature=2.0)
    student_logits = np.array([[1.0, 2.0, 0.5]], dtype=np.float32)
    teacher_logits = np.array([[1.2, 1.8, 0.4]], dtype=np.float32)

    loss = kd.compute_distillation_loss(student_logits, teacher_logits)
    assert loss >= 0.0
