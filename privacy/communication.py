"""
Module: privacy.communication

Purpose:
Provides serialization and deserialization helpers for encrypted model weights
and privacy payloads transferred across network interfaces.
"""


def serialize_update(update):
    """
    Serializes encrypted model weights for transmission over gRPC/HTTP.
    """
    return update


def deserialize_update(data):
    """
    Deserializes encrypted model updates received from clients/servers.
    """
    return data