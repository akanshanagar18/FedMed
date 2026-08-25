"""
Module: tests.unit.test_sla_compliance

Purpose:
Unit test suite for InstitutionalSLAAuditor (HIPAA/GDPR compliance checks, violations, and attestation certificates).
"""

import pytest
from governance.sla import InstitutionalSLAAuditor


def test_sla_auditor_passed_compliant():
    auditor = InstitutionalSLAAuditor(max_epsilon=10.0, min_participation_rate=0.75)
    audit = auditor.audit_compliance(
        run_id="run_001",
        epsilon_consumed=2.5,
        delta_consumed=1e-5,
        participating_nodes=["hospital_alpha", "hospital_beta", "hospital_gamma"],
        total_nodes=["hospital_alpha", "hospital_beta", "hospital_gamma"],
        avg_latency_ms=150.0,
        encryption_scheme="TenSEAL_CKKS",
    )
    assert audit["overall_compliant"] is True
    assert audit["status"] == "PASSED_COMPLIANT"
    assert len(audit["violations"]) == 0
    assert len(audit["certificate_hash"]) == 64


def test_sla_auditor_non_compliant_epsilon_exceeded():
    auditor = InstitutionalSLAAuditor(max_epsilon=5.0)
    audit = auditor.audit_compliance(
        run_id="run_002",
        epsilon_consumed=12.0,
        delta_consumed=1e-4,
        participating_nodes=["hospital_alpha", "hospital_beta"],
        total_nodes=["hospital_alpha", "hospital_beta"],
        avg_latency_ms=100.0,
    )
    assert audit["overall_compliant"] is False
    assert audit["status"] == "NON_COMPLIANT_VIOLATION"
    assert any("DP Privacy Budget Exceeded" in v for v in audit["violations"])
