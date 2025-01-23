import torch

def add_gaussian_noise(data: torch.Tensor, noise_level: float=0.02) -> torch.Tensor:
    """
    Add Gaussian noise to a 4D or 5D complex tensor.
    
    Parameters
    ----------
    data : torch.Tensor
        A complex tensor with dimensions [B, C, H, W] (4D) or [B, T, C, H, W] (5D).
    noise_level : float
        Level of noise to be added.
    
    Returns
    -------
    torch.Tensor
        Complex tensor with added noise.
    """
    # Ensure the input is a complex tensor
    if not torch.is_complex(data):
        raise ValueError("Input tensor must be a complex tensor.")

    # Compute maximum values for scaling noise
    max_real = data.real.amax(dim=(-1, -2, -3, -4), keepdim=True)
    max_imag = data.imag.amax(dim=(-1, -2, -3, -4), keepdim=True)

    # Generate Gaussian noise for real and imaginary components
    noise_real = noise_level * max_real * torch.randn_like(data.real)
    noise_imag = noise_level * max_imag * torch.randn_like(data.imag)

    # Add noise to real and imaginary parts
    noisy_data = data + (noise_real + 1j * noise_imag)

    return noisy_data