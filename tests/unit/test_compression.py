"""
Module: tests.unit.test_compression

Purpose:
Unit test suite for communication compression modules (Quantization, Sparsification, SignSGD).
"""

import numpy as np
import pytest

from communication.quantization import UniformQuantizer
from communication.sparsification import Sparsifier
from communication.sign_sgd import SignSGDCompressor


def test_uniform_quantizer_int8():
    quantizer = UniformQuantizer(bits=8)
    weights = [np.random.randn(10, 100).astype(np.float32)]

    payload = quantizer.compress(weights)
    assert payload.compression_ratio >= 3.0
    assert payload.bandwidth_savings_pct > 50.0

    decompressed = quantizer.decompress(payload)
    assert len(decompressed) == 1
    assert decompressed[0].shape == weights[0].shape
    np.testing.assert_allclose(decompressed[0], weights[0], atol=0.1)


def test_uniform_quantizer_int4():
    quantizer = UniformQuantizer(bits=4)
    weights = [np.random.randn(10, 100).astype(np.float32)]

    payload = quantizer.compress(weights)
    assert payload.compression_ratio >= 6.0

    decompressed = quantizer.decompress(payload)
    assert len(decompressed) == 1
    np.testing.assert_allclose(decompressed[0], weights[0], atol=0.5)


def test_topk_sparsifier():
    sparsifier = Sparsifier(ratio=0.1, method="topk")
    weights = [np.random.randn(10, 100).astype(np.float32)]

    payload = sparsifier.compress(weights)
    decompressed = sparsifier.decompress(payload)

    assert decompressed[0].shape == weights[0].shape
    assert payload.compression_ratio >= 3.0


def test_sign_sgd_compressor():
    compressor = SignSGDCompressor(use_error_feedback=True)
    weights = [np.random.randn(10, 100).astype(np.float32)]

    payload = compressor.compress(weights)
    assert payload.compressed_size_bytes < payload.original_size_bytes
    assert payload.compression_ratio >= 10.0

    decompressed = compressor.decompress(payload)
    assert len(decompressed) == 1
    assert decompressed[0].shape == weights[0].shape
