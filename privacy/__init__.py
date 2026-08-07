"""
Package: privacy

Purpose:
Privacy-Preserving Federated Learning Engine for FedMed v2.0.
Supports TenSEAL CKKS Homomorphic Encryption, Differential Privacy (Opacus), and Secure Aggregation.
"""

from privacy.context import create_ckks_context, get_public_context, serialize_context, deserialize_context
from privacy.encrypt import encrypt_weights_vector, encrypt_model_parameters
from privacy.decrypt import decrypt_weights_vector, decrypt_model_parameters
from privacy.aggregation import aggregate_encrypted_updates
from privacy.communication import serialize_encrypted_payload, deserialize_encrypted_payload

__all__ = [
    "create_ckks_context",
    "get_public_context",
    "serialize_context",
    "deserialize_context",
    "encrypt_weights_vector",
    "encrypt_model_parameters",
    "decrypt_weights_vector",
    "decrypt_model_parameters",
    "aggregate_encrypted_updates",
    "serialize_encrypted_payload",
    "deserialize_encrypted_payload",
]
