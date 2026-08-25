# Cryptographic Audit Trail & Dataset Lineage

## Overview
This document details cryptographic SHA-256 audit trail tracking and dataset lineage management implemented in **FedMed v2.0** (`audit/` and `data/versioning.py`).

---

## 1. Audit Trail Record Structure

Every experiment generates a signed audit record containing:
- `git_commit`: Immutable repository SHA commit hash.
- `dataset_hash`: SHA-256 dataset digest.
- `pipeline_hash`: SHA-256 MONAI transform preprocessing pipeline hash.
- `model_hash`: Combined architecture + hyperparameter hash.
- `audit_signature`: Signed cryptographic reproducibility hash.
