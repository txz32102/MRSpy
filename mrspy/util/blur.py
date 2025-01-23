import torch

def update_averaging(matrix_size1, matrix_size2, average: int=263):
    s0 = matrix_size1
    s1 = matrix_size2
    average_masks = torch.zeros((matrix_size1, matrix_size2))

    for k in range(-matrix_size1 // 2, matrix_size1 // 2):
        for l in range(-matrix_size2 // 2, matrix_size2 // 2):
            # Convert k and l to tensors before using in torch operations
            k_tensor = torch.tensor(k, dtype=torch.float32)
            l_tensor = torch.tensor(l, dtype=torch.float32)
            
            value = 1 + (average - 1) * 0.125 * (1 + torch.cos(2 * torch.pi * k_tensor / s0)) * (1 + torch.cos(2 * torch.pi * l_tensor / s1)) * 2
            average_masks[k + matrix_size1 // 2, l + matrix_size2 // 2] = torch.round(value)

    return average_masks