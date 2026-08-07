"""
Aggregation of encrypted model updates.
"""
from typing import Dict, List
from .exceptions import AggregationError, ShapeMismatchError
from .encryptor import EncryptedParam

def add_encrypted(param_a: EncryptedParam, param_b: EncryptedParam) -> EncryptedParam:
    if param_a["shape"] != param_b["shape"]:
        raise ShapeMismatchError(f"Shape mismatch during aggregation: {param_a['shape']} vs {param_b['shape']}")
    try:
        summed = param_a["data"] + param_b["data"]
        return {"data": summed, "shape": param_a["shape"]}
    except Exception as exc:
        raise AggregationError(f"Failed to add encrypted parameters: {exc}") from exc

def aggregate_encrypted_state_dicts(encrypted_state_dicts: List[Dict[str, EncryptedParam]]) -> Dict[str, EncryptedParam]:
    if not encrypted_state_dicts:
        raise AggregationError("No encrypted state dicts provided for aggregation")
    names = encrypted_state_dicts[0].keys()
    aggregated: Dict[str, EncryptedParam] = {}
    for name in names:
        try:
            accumulator = encrypted_state_dicts[0][name]
            for state_dict in encrypted_state_dicts[1:]:
                accumulator = add_encrypted(accumulator, state_dict[name])
            aggregated[name] = accumulator
        except KeyError as exc:
            raise AggregationError(f"Parameter '{name}' missing from one or more state dicts") from exc
    return aggregated

def weighted_aggregate_encrypted_state_dicts(encrypted_state_dicts: List[Dict[str, EncryptedParam]], weights: List[float]) -> Dict[str, EncryptedParam]:
    if len(encrypted_state_dicts) != len(weights):
        raise AggregationError(f"Number of state dicts and weights must match ({len(encrypted_state_dicts)} vs {len(weights)})")
    if not encrypted_state_dicts:
        raise AggregationError("No encrypted state dicts provided for aggregation")
    total_weight = sum(weights)
    if total_weight == 0:
        raise AggregationError("Sum of weights must not be zero")
    normalized_weights = [w / total_weight for w in weights]
    names = encrypted_state_dicts[0].keys()
    aggregated: Dict[str, EncryptedParam] = {}
    for name in names:
        accumulator = None
        for state_dict, weight in zip(encrypted_state_dicts, normalized_weights):
            try:
                param = state_dict[name]
                scaled = {"data": param["data"] * weight, "shape": param["shape"]}
            except KeyError as exc:
                raise AggregationError(f"Parameter '{name}' missing from one or more state dicts") from exc
            except Exception as exc:
                raise AggregationError(f"Failed to scale parameter '{name}': {exc}") from exc
            accumulator = scaled if accumulator is None else add_encrypted(accumulator, scaled)
        aggregated[name] = accumulator
    return aggregated
