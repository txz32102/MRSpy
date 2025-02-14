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


def extract_center_kspace(data: torch.Tensor, kspace_size: list, is_3d: bool = False):
    """
    Extract the central region of a tensor (2D or 3D).

    Args:
        data (torch.Tensor): The input tensor (2D or 3D).
        kspace_size (list): A list containing the size of the region to extract [rows, cols].
        is_3d (bool, optional): If True, assumes the input data is 3D (b, w, h). Default is False (2D).

    Returns:
        torch.Tensor: The extracted central region.
    """
    if is_3d:
        # For 3D input, batch size is the first dimension (b)
        b, w, h = data.size()

        # Compute center coordinates for 3D
        center_row = w // 2
        center_col = h // 2

        # Calculate start and end indices for rows and columns
        row_start = center_row - kspace_size[0] // 2
        row_end = center_row + kspace_size[0] // 2
        col_start = center_col - kspace_size[1] // 2
        col_end = center_col + kspace_size[1] // 2

        # Extract and return the central region for each sample in the batch
        return data[:, row_start:row_end, col_start:col_end]

    else:
        # For 2D input, calculate the center coordinates as before
        center_row = data.size(0) // 2
        center_col = data.size(1) // 2

        # Calculate start and end indices for rows and columns
        row_start = center_row - kspace_size[0] // 2
        row_end = center_row + kspace_size[0] // 2
        col_start = center_col - kspace_size[1] // 2
        col_end = center_col + kspace_size[1] // 2

        # Extract and return the central region
        return data[row_start:row_end, col_start:col_end]
