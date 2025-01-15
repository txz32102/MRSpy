from typing import Union
import scipy.io as sio
import numpy as np
import torch

def load_mrs_mat(
    file_path: str,
    variable_index: int = -1,
    output_type: str = "numpy"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Loads a .mat file and returns the specified variable as a NumPy array or PyTorch tensor.

    Args:
        file_path (str): Path to the .mat file.
        variable_index (int): Index of the variable to load. Default is -1 (last variable).
        output_type (str): Output type, either 'numpy' or 'tensor' (for PyTorch). Default is 'numpy'.

    Returns:
        Union[np.ndarray, torch.Tensor]: The loaded data as a NumPy array or PyTorch tensor.
    """
    mat_contents = sio.loadmat(file_path)
    variable_name = list(mat_contents.keys())[variable_index]
    data = mat_contents[variable_name]
    
    if output_type == "tensor":
        return torch.tensor(data)
    elif output_type == "numpy":
        return np.array(data)
    else:
        raise ValueError("output_type must be either 'numpy' or 'tensor'")