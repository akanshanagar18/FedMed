"""
Unit tests for privacy.tls_cert_gen (X.509 Certificate Generation & Loading).
"""

import os
import shutil
import pytest
from privacy.tls_cert_gen import ensure_tls_certificates, load_pem_bytes


def test_automatic_tls_certificate_generation(tmp_path):
    """Verify automatic generation of Root CA, Server, and Client certificates."""
    cert_dir = str(tmp_path / "certs")
    paths = ensure_tls_certificates(cert_dir)

    assert os.path.exists(paths["ca_cert"])
    assert os.path.exists(paths["ca_key"])
    assert os.path.exists(paths["server_cert"])
    assert os.path.exists(paths["server_key"])
    assert os.path.exists(paths["client_cert"])
    assert os.path.exists(paths["client_key"])

    ca_pem = load_pem_bytes(paths["ca_cert"])
    assert b"BEGIN CERTIFICATE" in ca_pem

    server_pem = load_pem_bytes(paths["server_cert"])
    assert b"BEGIN CERTIFICATE" in server_pem
