
from typing import Any, Optional
import pickle

try:
    
    from homophormic_encryption import serialization as he_serialization  
except Exception:
    he_serialization = None


def serialize_update(update: Any) -> bytes:
    
    if he_serialization is not None:
        
        if hasattr(he_serialization, "serialize_encrypted_state_dict"):
            return he_serialization.serialize_encrypted_state_dict(update)
       
        if hasattr(he_serialization, "serialize"):
            return he_serialization.serialize(update)
   
    return pickle.dumps(update)


def deserialize_update(data: bytes, context: Optional[Any] = None) -> Any:
    
    if he_serialization is not None:
        
        if hasattr(he_serialization, "deserialize_encrypted_state_dict"):
            if context is None:
                raise ValueError(
                    "deserialize_encrypted_state_dict requires a TenSEAL context; supply the server/public context when calling deserialize_update"
                )
            return he_serialization.deserialize_encrypted_state_dict(context, data)
        
        if hasattr(he_serialization, "deserialize"):
            return he_serialization.deserialize(data)
    
    return pickle.loads(data)