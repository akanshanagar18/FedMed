"""
Module: privacy.tls_cert_gen

Purpose:
Automatic Production-grade TLS X.509 Certificate Generator for FedMed v2.0.
Generates local Root CA, Server Certificate (with IP/DNS SANs), and Client Certificates
using Python `cryptography.x509` without requiring manual OpenSSL commands.
"""

import datetime
import ipaddress
import os
import logging
from typing import Dict, Tuple

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

logger = logging.getLogger(__name__)


def generate_rsa_private_key(key_size: int = 2048) -> rsa.RSAPrivateKey:
    """Generates an RSA private key."""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )


def save_pem_file(file_path: str, data: bytes) -> None:
    """Saves PEM bytes to disk, creating parent directories if needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(data)


def ensure_tls_certificates(
    cert_dir: str = "certs",
    valid_days: int = 365,
) -> Dict[str, str]:
    """
    Ensures Root CA, Server, and Client TLS certificates exist in `cert_dir`.
    Generates missing certificates automatically.
    
    Returns:
        Dict mapping cert names ('ca_cert', 'server_cert', 'server_key', 'client_cert', 'client_key') to absolute file paths.
    """
    cert_dir_abs = os.path.abspath(cert_dir)
    ca_cert_path = os.path.join(cert_dir_abs, "ca.crt")
    ca_key_path = os.path.join(cert_dir_abs, "ca.key")
    server_cert_path = os.path.join(cert_dir_abs, "server.crt")
    server_key_path = os.path.join(cert_dir_abs, "server.key")
    client_cert_path = os.path.join(cert_dir_abs, "client.crt")
    client_key_path = os.path.join(cert_dir_abs, "client.key")

    paths = {
        "ca_cert": ca_cert_path,
        "ca_key": ca_key_path,
        "server_cert": server_cert_path,
        "server_key": server_key_path,
        "client_cert": client_cert_path,
        "client_key": client_key_path,
    }

    # If all files exist, skip generation
    if all(os.path.exists(p) for p in paths.values()):
        logger.info(f"[TLS CERT GEN] Production TLS certificate suite verified at '{cert_dir_abs}'.")
        return paths

    logger.info(f"[TLS CERT GEN] Generating fresh X.509 production TLS certificate suite in '{cert_dir_abs}'...")

    now = datetime.datetime.now(datetime.timezone.utc)
    valid_until = now + datetime.timedelta(days=valid_days)

    # 1. Generate Root CA
    ca_key = generate_rsa_private_key()
    ca_name = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed Root Authority"),
        x509.NameAttribute(NameOID.COMMON_NAME, "FedMed Root CA"),
    ])

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(valid_until)
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    # Save CA
    save_pem_file(ca_key_path, ca_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    save_pem_file(ca_cert_path, ca_cert.public_bytes(serialization.Encoding.PEM))

    # 2. Generate Server Certificate (with SAN for localhost and IP addresses)
    server_key = generate_rsa_private_key()
    server_name = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed Central Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    san_extension = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ])

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(valid_until)
        .add_extension(san_extension, critical=False)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    save_pem_file(server_key_path, server_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    save_pem_file(server_cert_path, server_cert.public_bytes(serialization.Encoding.PEM))

    # 3. Generate Client Certificate
    client_key = generate_rsa_private_key()
    client_name = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed Hospital Node"),
        x509.NameAttribute(NameOID.COMMON_NAME, "fedmed-client"),
    ])

    client_cert = (
        x509.CertificateBuilder()
        .subject_name(client_name)
        .issuer_name(ca_name)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(valid_until)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    save_pem_file(client_key_path, client_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    save_pem_file(client_cert_path, client_cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"[TLS CERT GEN] Successfully generated X.509 certificate suite in '{cert_dir_abs}'.")
    return paths


def load_pem_bytes(file_path: str) -> bytes:
    """Reads a PEM certificate/key file from disk as bytes."""
    with open(file_path, "rb") as f:
        return f.read()
