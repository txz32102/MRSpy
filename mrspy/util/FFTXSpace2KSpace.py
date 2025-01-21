import torch

def fft_xspace_to_kspace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)

def FFTXSpace2KSpace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)