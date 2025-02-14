import torch

def fft_kspace_to_xspace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)

def FFTKSpace2XSpace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)

def apply_ifftshift_to_batch(batch, dim):
    # Apply ifftshift to each element in the batch
    return torch.stack([torch.fft.ifftshift(item, dim=dim) for item in batch])

def apply_fftshift_to_batch(batch, dim):
    # Apply fftshift to each element in the batch
    return torch.stack([torch.fft.fftshift(item, dim=dim) for item in batch])

def apply_fft_to_batch(batch, dim):
    # Apply ifft to each element in the batch
    return torch.stack([torch.fft.fft(item, dim=dim) for item in batch])

def fft_kspace_to_xspace_3d_batch(PreFFT, dim):
    return apply_fftshift_to_batch(apply_fft_to_batch(apply_ifftshift_to_batch(PreFFT, dim=dim), dim=dim), dim=dim)