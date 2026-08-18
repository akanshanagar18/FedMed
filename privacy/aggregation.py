
from typing import Dict, Iterable, List, Optional, Any

try:
    from homophormic_encryption import aggregation as he_aggregation  
    from homophormic_encryption import exceptions as he_exceptions  
except Exception:
    he_aggregation = None
    he_exceptions = None


class PrivacyAggregationError(Exception):
    pass


def aggregate_encrypted_updates(
    encrypted_updates: Iterable[Dict[str, Any]], weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    
    updates_list = list(encrypted_updates)
    if len(updates_list) == 0:
        raise ValueError("aggregate_encrypted_updates requires a non-empty iterable of updates")

    
    first = updates_list[0]
    if not isinstance(first, dict):
        raise ValueError("Each encrypted update must be a dict mapping parameter names to EncryptedParam-like objects")

    param_names = list(first.keys())
    if len(param_names) == 0:
        raise ValueError("Encrypted updates must contain at least one parameter")

    
    for i, upd in enumerate(updates_list[1:], start=1):
        if not isinstance(upd, dict):
            raise ValueError(f"Encrypted update at index {i} is not a dict")
        if list(upd.keys()) != param_names:
            raise ValueError(
                f"Parameter mismatch between updates: update 0 keys {param_names} vs update {i} keys {list(upd.keys())}"
            )

    
    if he_aggregation is None:
        raise RuntimeError(
            "Partner homomorphic_encryption.aggregation module is not available. "
            "Cannot perform encrypted aggregation on the server."
        )

    
    if hasattr(he_aggregation, "aggregate_encrypted_state_dicts"):
        try:
            if weights is not None:
                if not hasattr(he_aggregation, "weighted_aggregate_encrypted_state_dicts"):
                    
                    raise RuntimeError(
                        "Partner aggregation module does not expose weighted_aggregate_encrypted_state_dicts; "
                        "weighted aggregation is unsupported server-side. Consider applying weights client-side."
                    )
         
                return he_aggregation.weighted_aggregate_encrypted_state_dicts(updates_list, weights)
            
            return he_aggregation.aggregate_encrypted_state_dicts(updates_list)
        except Exception as exc:
            
            if he_exceptions is not None:
                
                shape_exc = getattr(he_exceptions, "ShapeMismatchError", None)
                agg_exc = getattr(he_exceptions, "AggregationError", None)
                if shape_exc is not None and isinstance(exc, shape_exc):
                    raise PrivacyAggregationError(f"Shape mismatch during aggregation: {exc}") from exc
                if agg_exc is not None and isinstance(exc, agg_exc):
                    raise PrivacyAggregationError(f"Low-level aggregation failed: {exc}") from exc
            raise PrivacyAggregationError(f"Low-level aggregation failed: {exc}") from exc

    
    if hasattr(he_aggregation, "aggregate_ciphertexts"):
        aggregated: Dict[str, Any] = {}
        try:
            for name in param_names:
                ciphertexts = [upd[name] for upd in updates_list]
                try:
                    agg = he_aggregation.aggregate_ciphertexts(ciphertexts, weights=weights)
                except TypeError:
                    agg = he_aggregation.aggregate_ciphertexts(ciphertexts)
                aggregated[name] = agg
            return aggregated
        except Exception as exc:
            raise PrivacyAggregationError(f"Low-level aggregate_ciphertexts failed: {exc}") from exc

    
    raise RuntimeError(
        "Partner homomorphic_encryption.aggregation module does not expose a compatible aggregation API. "
        "Expected one of: aggregate_encrypted_state_dicts, weighted_aggregate_encrypted_state_dicts, or aggregate_ciphertexts."
    )