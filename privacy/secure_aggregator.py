from typing import List, Optional, Any
from . import aggregation as privacy_aggregation
from . import communication as privacy_comm

class SecureAggregator:
    def __init__(self, context: Optional[Any] = None) -> None:
        self.context = context

    def aggregate(self, serialized_updates: List[bytes], weights: Optional[List[float]] = None) -> bytes:
        if not serialized_updates:
            raise ValueError("No updates provided to SecureAggregator.aggregate")
        deserialized = [privacy_comm.deserialize_update(b, context=self.context) for b in serialized_updates]
        aggregated = privacy_aggregation.aggregate_encrypted_updates(deserialized, weights=weights)
        serialized = privacy_comm.serialize_update(aggregated)
        return serialized