"""
Encryption of PyTorch model parameters using CKKS.
"""
from typing import Dict
import tenseal as ts
import torch
from .tensor_utils import tensor_to_list
from .exceptions import EncryptionError

EncryptedParam = Dict[str, object]

def encrypt_parameter(context: ts.Context, tensor: torch.Tensor) -> EncryptedParam:
    try:
        values, shape = tensor_to_list(tensor)
        encrypted_vector = ts.ckks_vector(context, values)
        return {"data": encrypted_vector, "shape": shape}
    except Exception as exc:
        raise EncryptionError(f"Failed to encrypt parameter: {exc}") from exc

def encrypt_state_dict(context: ts.Context, state_dict: Dict[str, torch.Tensor]) -> Dict[str, EncryptedParam]:
    encrypted_state = {}
    for name, tensor in state_dict.items():
        try:
            encrypted_state[name] = encrypt_parameter(context, tensor)
        except EncryptionError as exc:
            raise EncryptionError(f"Failed to encrypt parameter '{name}': {exc}") from exc
    return encrypted_state
