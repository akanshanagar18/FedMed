"""
Module: tests.unit.test_benchmark_visualizer

Purpose:
Unit tests for BenchmarkVisualizer figure generation.
"""

import os
import pytest
from utils.benchmark_visualizer import BenchmarkVisualizer


def test_benchmark_visualizer_generates_10_figures(tmp_path):
    output_dir = str(tmp_path / "benchmark_run")
    visualizer = BenchmarkVisualizer(output_dir=output_dir)

    plot_paths = visualizer.generate_all_plots({"benchmark_id": "test_vis", "results": []})
    assert len(plot_paths) == 20  # 10 PNGs + 10 PDFs

    for p in plot_paths:
        assert os.path.exists(p)
