"""
Module: tests.unit.test_byzantine_security

Purpose:
Unit test suite for Byzantine aggregators (Krum, MultiKrum, TrimmedMean, Median, Bulyan, FLTrust) and attack simulators.
"""

import numpy as np
import pytest

from security.byzantine import (
    KrumAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator,
    MedianAggregator,
    BulyanAggregator,
    FLTrustAggregator,
)
from security.attacks import (
    LabelFlippingAttack,
    ModelPoisoningAttack,
    GradientPoisoningAttack,
    BackdoorAttack,
    SybilAttack,
)


def test_krum_and_multikrum():
    u1 = [np.array([1.0, 1.0], dtype=np.float32)]
    u2 = [np.array([1.1, 0.9], dtype=np.float32)]
    u3 = [np.array([0.9, 1.1], dtype=np.float32)]
    u4 = [np.array([1.05, 0.95], dtype=np.float32)]
    u_bad = [np.array([100.0, 100.0], dtype=np.float32)]

    updates = [u1, u2, u3, u4, u_bad]


    krum = KrumAggregator(f_byzantine=1)
    res_krum = krum.aggregate(updates)
    assert np.max(res_krum[0]) < 10.0

    mkrum = MultiKrumAggregator(f_byzantine=1, m_select=2)
    res_mkrum = mkrum.aggregate(updates)
    assert np.max(res_mkrum[0]) < 10.0


def test_trimmed_mean_and_median():
    u1 = [np.array([1.0], dtype=np.float32)]
    u2 = [np.array([2.0], dtype=np.float32)]
    u3 = [np.array([3.0], dtype=np.float32)]
    u_bad = [np.array([100.0], dtype=np.float32)]

    updates = [u1, u2, u3, u_bad]

    tmean = TrimmedMeanAggregator(beta=0.25)
    res_tmean = tmean.aggregate(updates)
    assert res_tmean[0][0] < 50.0

    med = MedianAggregator()
    res_med = med.aggregate(updates)
    assert res_med[0][0] <= 3.0


def test_fltrust_aggregator():
    root_u = [np.array([1.0, 1.0], dtype=np.float32)]
    u1 = [np.array([1.1, 0.9], dtype=np.float32)]
    u_bad = [np.array([-10.0, -10.0], dtype=np.float32)]

    fltrust = FLTrustAggregator(server_root_update=root_u)
    res = fltrust.aggregate([u1, u_bad])
    assert res[0][0] > 0.0


def test_attacks():
    lf = LabelFlippingAttack(target_class=0, flipped_class=1)
    labels = np.array([0, 1, 0, 2])
    flipped = lf.flip_labels(labels)
    assert flipped[0] == 1

    mp = ModelPoisoningAttack(attack_type="sign_flip")
    poisoned = mp.poison_weights([np.array([1.0, 2.0], dtype=np.float32)])
    assert poisoned[0][0] < 0.0
