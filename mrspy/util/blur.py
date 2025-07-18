import torch

def update_averaging(matrix_size1, matrix_size2, average: int=263, average_mode: str='Weighted'):
    s0 = matrix_size1
    s1 = matrix_size2
    average_masks = torch.zeros((matrix_size1, matrix_size2))
    average_list = []

    if average_mode == 'Weighted':
        for k in range(-matrix_size1 // 2, matrix_size1 // 2):
            for l in range(-matrix_size2 // 2, matrix_size2 // 2):
                k_tensor = torch.tensor(k, dtype=torch.float32)
                l_tensor = torch.tensor(l, dtype=torch.float32)

                value = 1 + (average - 1) * 0.25 * (1 + torch.cos(2 * torch.pi * k_tensor / s0)) * (1 + torch.cos(2 * torch.pi * l_tensor / s1))
                rounded_value = torch.round(value)
                average_masks[k + matrix_size1 // 2, l + matrix_size2 // 2] = rounded_value
                average_list.append(int(rounded_value.item()))

        average_list_sum = sum(average_list)
    else:
        average_masks.fill(average)
        average_list = [average] * (matrix_size1 * matrix_size2)
        average_list_sum = average * matrix_size1 * matrix_size2

    return average_masks, average_list_sum, average_list
