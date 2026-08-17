from typing import Dict, List

from homophormic_encryption.aggregation import (
    aggregate_encrypted_state_dicts,
    weighted_aggregate_encrypted_state_dicts,
)
from homophormic_encryption.encryptor import EncryptedParam
from homophormic_encryption.exceptions import (
    AggregationError,
    ShapeMismatchError,
)


def validate_encrypted_updates(
    encrypted_updates: List[Dict[str, EncryptedParam]]
) -> None:
    if not encrypted_updates:
        raise AggregationError("No encrypted updates provided")

    reference_names = set(encrypted_updates[0].keys())

    for index, update in enumerate(encrypted_updates):
        if set(update.keys()) != reference_names:
            raise AggregationError(
                f"Parameter mismatch in encrypted update {index}"
            )

        for name in reference_names:
            reference_shape = encrypted_updates[0][name]["shape"]
            current_shape = update[name]["shape"]

            if current_shape != reference_shape:
                raise ShapeMismatchError(
                    f"Shape mismatch for parameter '{name}': "
                    f"{reference_shape} vs {current_shape}"
                )


def validate_weights(
    encrypted_updates: List[Dict[str, EncryptedParam]],
    weights: List[float],
) -> None:
    if len(encrypted_updates) != len(weights):
        raise AggregationError(
            f"Number of updates and weights must match "
            f"({len(encrypted_updates)} vs {len(weights)})"
        )

    if not weights:
        raise AggregationError("No aggregation weights provided")

    if any(weight < 0 for weight in weights):
        raise AggregationError("Aggregation weights must not be negative")

    if sum(weights) == 0:
        raise AggregationError(
            "Sum of aggregation weights must not be zero"
        )


def secure_aggregate(
    encrypted_updates: List[Dict[str, EncryptedParam]],
) -> Dict[str, EncryptedParam]:
    validate_encrypted_updates(encrypted_updates)

    return aggregate_encrypted_state_dicts(encrypted_updates)


def secure_weighted_aggregate(
    encrypted_updates: List[Dict[str, EncryptedParam]],
    weights: List[float],
) -> Dict[str, EncryptedParam]:
    validate_encrypted_updates(encrypted_updates)
    validate_weights(encrypted_updates, weights)

    return weighted_aggregate_encrypted_state_dicts(
        encrypted_updates,
        weights,
    )