import torch
from typing import List, Tuple

def tensor_to_list(tensor: torch.Tensor) -> Tuple[List[float], Tuple[int, ...]]:
    return tensor.flatten().tolist(), tuple(tensor.shape)

def list_to_tensor(values: List[float], shape: Tuple[int, ...], dtype=None) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=dtype)
    return tensor.reshape(shape)
