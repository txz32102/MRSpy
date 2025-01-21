import torch

def center_crop_kspace(kspace_data, crop_size):
    # Assuming kspace_data is a square 2D tensor 
    # and crop_size is the size of one dimension of the square crop
    center = kspace_data.shape[0] // 2
    half_crop = crop_size // 2
    
    # Define the indices for cropping
    start = center - half_crop
    end = center + half_crop
    
    # Crop the k-space data
    return kspace_data[start:end, start:end]


def extract_center_kspace(data: torch.Tensor, kspace_size: list):
    """
    Extract the central region of a 2D tensor.

    Args:
        data (torch.Tensor): The input 2D tensor.
        kspace_size (list): A list containing the size of the region to extract [rows, cols].

    Returns:
        torch.Tensor: The extracted central region.
    """
    # Compute center coordinates
    center_row = data.size(0) // 2
    center_col = data.size(1) // 2

    # Calculate start and end indices for rows and columns
    row_start = center_row - kspace_size[0] // 2 - 1
    row_end = center_row + kspace_size[0] // 2 - 1
    col_start = center_col - kspace_size[1] // 2 - 1
    col_end = center_col + kspace_size[1] // 2 - 1

    # Extract and return the central region
    return data[row_start:row_end, col_start:col_end]