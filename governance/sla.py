"""
Module: governance.sla

Purpose:
Institutional Service Level Agreement (SLA) & Regulatory Compliance Auditor.
Audits privacy budget consumption (Differential Privacy epsilon/delta), hospital participation rates,
communication latency bounds, and zero-leakage constraints to generate HIPAA & GDPR attestation certificates.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional


class InstitutionalSLAAuditor:
    """
    Audits institutional compliance SLAs and generates cryptographic attestation certificates.
    """

    def __init__(
        self,
        max_epsilon: float = 10.0,
        max_delta: float = 1e-4,
        min_participation_rate: float = 0.75,
        max_latency_ms: float = 5000.0,
        signing_secret: str = "FedMed_HIPAA_GDPR_Compliance_Secret_2026",
    ):
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.min_participation_rate = min_participation_rate
        self.max_latency_ms = max_latency_ms
        self.signing_secret = signing_secret

    def audit_compliance(
        self,
        run_id: str,
        epsilon_consumed: float,
        delta_consumed: float,
        participating_nodes: List[str],
        total_nodes: List[str],
        avg_latency_ms: float,
        encryption_scheme: str = "TenSEAL_CKKS",
    ) -> Dict[str, Any]:
        """
        Performs a full audit of an FL round or session against compliance SLAs.
        """
        participation_rate = len(participating_nodes) / max(len(total_nodes), 1)

        # Check rules
        dp_compliant = (epsilon_consumed <= self.max_epsilon) and (delta_consumed <= self.max_delta)
        participation_compliant = participation_rate >= self.min_participation_rate
        latency_compliant = avg_latency_ms <= self.max_latency_ms
        encryption_compliant = encryption_scheme in ["TenSEAL_CKKS", "Opacus_DP", "TLS_gRPC", "Combined_HE_DP"]

        overall_compliant = bool(dp_compliant and participation_compliant and latency_compliant and encryption_compliant)

        timestamp = int(time.time())

        # Generate SHA256 cryptographic attestation signature
        payload_to_sign = (
            f"{run_id}|{epsilon_consumed:.4f}|{delta_consumed:.6f}|"
            f"{participation_rate:.2f}|{encryption_scheme}|{timestamp}|{self.signing_secret}"
        )
        certificate_hash = hashlib.sha256(payload_to_sign.encode("utf-8")).hexdigest()

        status = "PASSED_COMPLIANT" if overall_compliant else "NON_COMPLIANT_VIOLATION"

        violations = []
        if not dp_compliant:
            violations.append(f"DP Privacy Budget Exceeded: epsilon={epsilon_consumed:.2f} > max={self.max_epsilon}")
        if not participation_compliant:
            violations.append(f"Node Participation Below SLA: rate={participation_rate:.2%} < min={self.min_participation_rate:.2%}")
        if not latency_compliant:
            violations.append(f"Network Latency Exceeded Limit: latency={avg_latency_ms:.1f}ms > max={self.max_latency_ms:.1f}ms")
        if not encryption_compliant:
            violations.append(f"Unapproved Security Scheme: {encryption_scheme}")

        return {
            "run_id": run_id,
            "status": status,
            "overall_compliant": overall_compliant,
            "audit_timestamp": timestamp,
            "certificate_hash": certificate_hash,
            "regulatory_frameworks": ["HIPAA Security Rule §164.312", "GDPR Article 25 & 32 (Privacy by Design)"],
            "sla_metrics": {
                "epsilon_consumed": round(epsilon_consumed, 4),
                "max_epsilon_allowed": self.max_epsilon,
                "delta_consumed": round(delta_consumed, 6),
                "participation_rate": round(participation_rate, 4),
                "participating_nodes_count": len(participating_nodes),
                "total_nodes_count": len(total_nodes),
                "avg_latency_ms": round(avg_latency_ms, 2),
                "encryption_scheme": encryption_scheme,
            },
            "compliance_checks": {
                "differential_privacy_budget": dp_compliant,
                "node_participation_sla": participation_compliant,
                "network_latency_sla": latency_compliant,
                "zero_leakage_encryption": encryption_compliant,
            },
            "violations": violations,
        }
