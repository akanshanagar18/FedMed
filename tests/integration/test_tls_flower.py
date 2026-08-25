"""
Integration test verifying TLS certificate suite loading and configuration.
"""

import os
import pytest
from privacy.tls_cert_gen import ensure_tls_certificates, load_pem_bytes
from configs.loader import load_config


def test_tls_config_loading(tmp_path):
    """Verify configs/tls.yaml loads TLS configuration cleanly."""
    tls_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "tls.yaml")
    assert os.path.exists(tls_config_path)

    app_cfg = load_config(tls_config_path)
    assert app_cfg.tls.enabled is True
    assert app_cfg.tls.verify_server is True

    certs = ensure_tls_certificates(str(tmp_path / "certs"))
    ca_bytes = load_pem_bytes(certs["ca_cert"])
    server_bytes = load_pem_bytes(certs["server_cert"])
    key_bytes = load_pem_bytes(certs["server_key"])

    assert len(ca_bytes) > 0
    assert len(server_bytes) > 0
    assert len(key_bytes) > 0
