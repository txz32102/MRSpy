from typing import Union
import scipy.io as sio
import numpy as np
import torch
import h5py

def load_toy_data(batch_size: int = None, DceNum: int = 32, spec_len: int = 120, image_size: int = 32, reshape: bool = False) -> torch.Tensor:
    if(batch_size is not None):
        if(reshape):
            return torch.randn(batch_size, DceNum, spec_len, image_size, image_size).reshape(batch_size, DceNum * spec_len, image_size, image_size)
        return torch.randn(batch_size, DceNum, spec_len, image_size, image_size)
    if(reshape):
        return torch.randn(DceNum, spec_len, image_size, image_size).reshape(DceNum * spec_len, image_size, image_size)
    return torch.randn(DceNum, spec_len, image_size, image_size)

def load_mrs_mat(
    file_path: str,
    variable_index: int = -1,
    output_type: str = "numpy",
    dtype: Union[str, torch.dtype] = torch.float32,
    device: str = "cpu",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Loads a .mat file and returns the specified variable as a NumPy array or PyTorch tensor.

    Tries to load with scipy.io first, and falls back to h5py if there's an error.

    Args:
        file_path (str): Path to the .mat file.
        variable_index (int): Index of the variable to load. Default is -1 (last variable).
        output_type (str): Output type, either 'numpy' or 'tensor' (for PyTorch). Default is 'numpy'.
        dtype (Union[str, torch.dtype]): Desired data type for the output. Default is torch.float32.
        device (str): Device to load the data onto if output_type is 'tensor'. Default is 'cpu'.

    Returns:
        Union[np.ndarray, torch.Tensor]: The loaded data as a NumPy array or PyTorch tensor.

    Raises:
        ValueError: If output_type is not 'numpy' or 'tensor'.
    """
    try:
        # First attempt with scipy.io
        mat_contents = sio.loadmat(file_path)
        variable_name = list(mat_contents.keys())[variable_index]
        data = mat_contents[variable_name]
    except Exception as e:
        # If scipy.io fails, try with h5py
        with h5py.File(file_path, 'r') as f:
            variable_name = list(f.keys())[variable_index]
            data = f[variable_name][:]
            permute_order = tuple(reversed(range(data.ndim)))
            data = np.transpose(data, permute_order)
    
    # Convert the data to the specified type and device
    if output_type == "tensor":
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)  # Convert string to torch.dtype
        tensor = torch.tensor(data, dtype=dtype, device=device)
        return tensor
    elif output_type == "numpy":
        return np.array(data, dtype=dtype)
    else:
        raise ValueError("output_type must be either 'numpy' or 'tensor'")