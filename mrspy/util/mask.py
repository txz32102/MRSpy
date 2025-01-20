from typing import Union
import torch

def mask_tensor(
    tensor: torch.Tensor, 
    dim: int, 
    value: Union[int, float] = 0, 
    num_to_mask: int = 0
) -> torch.Tensor:
  """
  Masks a specified dimension of a PyTorch tensor with a given value.

  Args:
    tensor: The input tensor.
    dim: The dimension to mask.
    value: The value to use for masking (default: 0).
    num_to_mask: The number of elements to mask along the specified dimension (default: 0).

  Returns:
    The masked tensor.
  """

  if num_to_mask == 0:
    return tensor

  dim_size = tensor.size(dim)
  start_idx = dim_size // 2 - num_to_mask // 2 
  end_idx = start_idx + num_to_mask

  mask = torch.ones_like(tensor)
  mask[:, start_idx:end_idx, :, :] = value 
  masked_tensor = tensor * mask

  return masked_tensor