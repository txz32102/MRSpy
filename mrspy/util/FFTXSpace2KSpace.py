import torch

def FFTXSpace2KSpace(PreFFT, dim):
    return torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(PreFFT, dim=dim), dim=dim), dim=dim)