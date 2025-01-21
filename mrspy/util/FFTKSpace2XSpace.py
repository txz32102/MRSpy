import torch

def fft_kspace_to_xspace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)

def FFTKSpace2XSpace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)