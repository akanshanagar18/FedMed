
from typing import Any
import pickle

try:
    
    from homophormic_encryption import serialization as he_serialization  
except Exception:
    he_serialization = None


def serialize_update(update: Any) -> bytes:
    
    if he_serialization is not None and hasattr(he_serialization, "serialize"):
        return he_serialization.serialize(update)
    return pickle.dumps(update)


def deserialize_update(data: bytes) -> Any:
   
    if he_serialization is not None and hasattr(he_serialization, "deserialize"):
        return he_serialization.deserialize(data)
    return pickle.loads(data)