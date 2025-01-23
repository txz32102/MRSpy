import torch

def check_real_nd(tensor, idx1, idx2, idx3=None, idx4=None):
    dimensions = tensor.dim()
    
    # Check if the input is 2D, 3D, or 4D
    if dimensions < 2 or dimensions > 4:
        raise ValueError('The input must be a 2D, 3D, or 4D tensor.')
    
    # Calculate mean and standard deviation of the tensor
    mean_val = torch.mean(tensor)
    std_val = torch.std(tensor)
    
    # Print mean and std in .10f format
    print(f'Mean of the tensor: {mean_val:.10f}')
    print(f'Standard deviation of the tensor: {std_val:.10f}')
    
    # Check and print value based on dimensions
    if dimensions == 2:
        if idx1 >= tensor.size(0) or idx2 >= tensor.size(1) or idx1 < 0 or idx2 < 0:
            raise IndexError('Index out of bounds for 2D tensor.')
        print(f'Value at ({idx1}, {idx2}): {tensor[idx1, idx2]:.10f}')
    elif dimensions == 3:
        if idx3 is None or idx1 >= tensor.size(0) or idx2 >= tensor.size(1) or idx3 >= tensor.size(2) or idx1 < 0 or idx2 < 0 or idx3 < 0:
            raise IndexError('Index out of bounds for 3D tensor.')
        print(f'Value at ({idx1}, {idx2}, {idx3}): {tensor[idx1, idx2, idx3]:.10f}')
    else:  # 4D
        if idx4 is None or idx1 >= tensor.size(0) or idx2 >= tensor.size(1) or idx3 >= tensor.size(2) or idx4 >= tensor.size(3) or idx1 < 0 or idx2 < 0 or idx3 < 0 or idx4 < 0:
            raise IndexError('Index out of bounds for 4D tensor.')
        print(f'Value at ({idx1}, {idx2}, {idx3}, {idx4}): {tensor[idx1, idx2, idx3, idx4]:.10f}')

import torch

def check_img_nd(img, idx1, idx2, idx3=None, idx4=None):
    # Determine the number of dimensions in img
    dims = img.dim()
    
    # Real and Imaginary parts
    real_part = img.real
    imag_part = img.imag

    # Print max, min, mean, and std of real part
    print('Real part:')
    print(f'Max: {torch.max(real_part).item():.10f}')
    print(f'Min: {torch.min(real_part).item():.10f}')
    print(f'Mean: {torch.mean(real_part).item():.10f}')
    print(f'Std: {torch.std(real_part).item():.10f}')

    # Print max, min, mean, and std of imaginary part
    print('Imaginary part:')
    print(f'Max: {torch.max(imag_part).item():.10f}')
    print(f'Min: {torch.min(imag_part).item():.10f}')
    print(f'Mean: {torch.mean(imag_part).item():.10f}')
    print(f'Std: {torch.std(imag_part).item():.10f}')

    # Indexing based on number of dimensions
    if dims == 2:
        if idx3 is not None or idx4 is not None:
            raise ValueError('Too many indices for 2D tensor')
        value = img[idx1, idx2]
    elif dims == 3:
        if idx4 is not None:
            raise ValueError('Too many indices for 3D tensor')
        value = img[idx1, idx2, idx3]
    elif dims == 4:
        value = img[idx1, idx2, idx3, idx4]
    else:
        raise ValueError('Input must be 2D, 3D, or 4D tensor')

    # Print the value at the given index
    print(f'Value at index real: {value.real.item():.10f}')
    print(f'Value at index imag: {value.imag.item():.10f}')