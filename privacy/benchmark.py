import time
from typing import List, Optional, Any
from . import communication as privacy_comm
from .secure_aggregator import SecureAggregator

def time_serialization_roundtrip(update: Any, rounds: int = 10) -> dict:
    t0 = time.perf_counter()
    for _ in range(rounds):
        b = privacy_comm.serialize_update(update)
        _ = privacy_comm.deserialize_update(b)
    t1 = time.perf_counter()
    return {"rounds": rounds, "total_seconds": t1 - t0, "per_round": (t1 - t0) / rounds}

def time_aggregation(serialized_updates: List[bytes], weights: Optional[List[float]] = None) -> dict:
    agg = SecureAggregator()
    t0 = time.perf_counter()
    result = agg.aggregate(serialized_updates, weights=weights)
    t1 = time.perf_counter()
    return {"aggregate_seconds": t1 - t0, "result_size_bytes": len(result)}