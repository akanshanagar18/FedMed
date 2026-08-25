"""
Module: tests.unit.test_executive_reports

Purpose:
Unit test suite for ExecutiveAnalyticsEngine (generating KPI summary, risk overview, markdown report).
"""

import pytest
from analytics.executive_reports import ExecutiveAnalyticsEngine


def test_executive_reports_summary_generation():
    engine = ExecutiveAnalyticsEngine()
    report = engine.generate_executive_summary(
        total_rounds_executed=30,
        avg_dice_score=0.87,
        hospital_participation_rate=1.0,
    )
    assert "kpis" in report
    assert report["kpis"]["mean_segmentation_dice"] == 0.87
    assert "risk_overview" in report
    assert "markdown_report" in report
    assert "FedMed v2.0" in report["markdown_report"]
