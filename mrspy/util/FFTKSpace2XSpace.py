import torch

def FFTKSpace2XSpace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)