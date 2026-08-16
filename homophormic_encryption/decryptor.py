"""
Decryption of CKKS-encrypted model parameters back into PyTorch tensors.
"""
from typing import Dict, Optional
import torch
from .tensor_utils import list_to_tensor
from .exceptions import DecryptionError
from .encryptor import EncryptedParam

def decrypt_parameter(encrypted_param: EncryptedParam, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    try:
        values = encrypted_param["data"].decrypt()
        shape = encrypted_param["shape"]
    except Exception as exc:
        raise DecryptionError(f"Failed to decrypt parameter: {exc}") from exc
    return list_to_tensor(values, shape, dtype=dtype) if dtype else list_to_tensor(values, shape)

def decrypt_state_dict(encrypted_state_dict: Dict[str, EncryptedParam], dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    decrypted = {}
    for name, encrypted_param in encrypted_state_dict.items():
        try:
            decrypted[name] = decrypt_parameter(encrypted_param, dtype=dtype)
        except DecryptionError as exc:
            raise DecryptionError(f"Failed to decrypt parameter '{name}': {exc}") from exc
    return decrypted
