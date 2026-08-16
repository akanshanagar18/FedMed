from typing import Dict, List

from homomorphic_encryption.aggregation import (
    aggregate_encrypted_state_dicts,
    weighted_aggregate_encrypted_state_dicts,
)
from homomorphic_encryption.encryptor import EncryptedParam


def aggregate_encrypted_updates(
    encrypted_updates: List[Dict[str, EncryptedParam]]
) -> Dict[str, EncryptedParam]:
    """
    Aggregate encrypted model updates without decrypting them.

    Args:
        encrypted_updates: Encrypted model updates received from clients.

    Returns:
        Aggregated encrypted model update.
    """
    return aggregate_encrypted_state_dicts(encrypted_updates)


def weighted_aggregate_encrypted_updates(
    encrypted_updates: List[Dict[str, EncryptedParam]],
    weights: List[float]
) -> Dict[str, EncryptedParam]:
    """
    Perform weighted aggregation of encrypted model updates.

    Args:
        encrypted_updates: Encrypted model updates received from clients.
        weights: Weight associated with each client update.

    Returns:
        Weighted aggregated encrypted model update.
    """
    return weighted_aggregate_encrypted_state_dicts(
        encrypted_updates,
        weights
    )
