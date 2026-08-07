"""
Module: privacy.communication

Purpose:
Serialization and deserialization helpers for homomorphic encryption payloads,
context bytes, and ciphertext streams transferred over gRPC and REST interfaces.
"""

import base64
import json
from typing import Any, Dict


def serialize_encrypted_payload(payload: Dict[str, Any]) -> str:
    """
    Serializes an encrypted update dictionary into a base64 JSON string.
    """
    serializable = {
        "encrypted_chunks": [
            [base64.b64encode(chunk).decode("ascii") for chunk in layer_chunks]
            for layer_chunks in payload["encrypted_chunks"]
        ],
        "shapes": payload.get("shapes", []),
        "encryption_time_ms": payload.get("encryption_time_ms", 0.0),
        "ciphertext_size_bytes": payload.get("ciphertext_size_bytes", 0),
    }
    return json.dumps(serializable)


def deserialize_encrypted_payload(data_str: str) -> Dict[str, Any]:
    """
    Deserializes a base64 JSON string back into encrypted update dictionary.
    """
    raw = json.loads(data_str)
    encrypted_chunks = [
        [base64.b64decode(chunk_b64.encode("ascii")) for chunk_b64 in layer_chunks]
        for layer_chunks in raw["encrypted_chunks"]
    ]
    return {
        "encrypted_chunks": encrypted_chunks,
        "shapes": [tuple(s) for s in raw.get("shapes", [])],
        "encryption_time_ms": raw.get("encryption_time_ms", 0.0),
        "ciphertext_size_bytes": raw.get("ciphertext_size_bytes", 0),
    }


def serialize_update(update: Any) -> Any:
    if isinstance(update, dict) and "encrypted_chunks" in update:
        return serialize_encrypted_payload(update)
    return update


def deserialize_update(data: Any) -> Any:
    if isinstance(data, str) and "encrypted_chunks" in data:
        return deserialize_encrypted_payload(data)
    return data