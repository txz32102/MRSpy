import torch

def fft_xspace_to_kspace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)

def FFTXSpace2KSpace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)

def apply_ifftshift_to_batch(batch, dim):
    # Apply ifftshift to each element in the batch
    return torch.stack([torch.fft.ifftshift(item, dim=dim) for item in batch])

def apply_fftshift_to_batch(batch, dim):
    # Apply fftshift to each element in the batch
    return torch.stack([torch.fft.fftshift(item, dim=dim) for item in batch])

def apply_ifft_to_batch(batch, dim):
    # Apply ifft to each element in the batch
    return torch.stack([torch.fft.ifft(item, dim=dim) for item in batch])

def fft_xspace_to_kspace_3d_batch(PreFFT, dim):
    return apply_fftshift_to_batch(apply_ifft_to_batch(apply_ifftshift_to_batch(PreFFT, dim=dim), dim=dim), dim=dim)