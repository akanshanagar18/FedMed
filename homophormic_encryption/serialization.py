"""
Serialization of encrypted state dictionaries for network transfer.
"""
import pickle
import tenseal as ts
from typing import Dict

from .exceptions import EncryptionError
from .encryptor import EncryptedParam

def serialize_encrypted_state_dict(encrypted_state_dict: Dict[str, EncryptedParam]) -> bytes:
    """Serialize an encrypted state dict into bytes for network transfer."""
    serializable_dict = {}
    for name, param in encrypted_state_dict.items():
        try:
            serializable_dict[name] = {
                "data": param["data"].serialize(),
                "shape": param["shape"]
            }
        except Exception as exc:
            raise EncryptionError(f"Failed to serialize parameter '{name}': {exc}") from exc
    return pickle.dumps(serializable_dict)

def deserialize_encrypted_state_dict(context: ts.Context, data: bytes) -> Dict[str, EncryptedParam]:
    """Deserialize bytes back into an encrypted state dict."""
    try:
        loaded_dict = pickle.loads(data)
    except Exception as exc:
        raise EncryptionError(f"Failed to unpickle data: {exc}") from exc
    
    encrypted_state_dict = {}
    for name, param in loaded_dict.items():
        try:
            encrypted_vector = ts.ckks_vector_from(context, param["data"])
            encrypted_state_dict[name] = {
                "data": encrypted_vector,
                "shape": param["shape"]
            }
        except Exception as exc:
            raise EncryptionError(f"Failed to deserialize parameter '{name}': {exc}") from exc
    return encrypted_state_dict
