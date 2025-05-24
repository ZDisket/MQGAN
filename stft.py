import torch
import torch.nn as nn # For inheritance from nn.Module
import torch.nn.functional as F # Often useful alongside STFT, though not directly used in TorchSTFT

class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        # Note: The original 'window' argument was a string, but torch.hann_window is used directly.
        # If other window types were intended, this would need modification.
        # For simplicity and matching the provided code, we'll assume 'hann' is fixed.
        self.window = torch.hann_window(win_length)

    def transform(self, input_data):
        # Ensure window is on the same device as input_data
        self.window = self.window.to(input_data.device)

        forward_transform = torch.stft(
            input_data,
            n_fft=self.filter_length, # n_fft parameter for torch.stft
            hop_length=self.hop_length,
            win_length=self.win_length, # win_length parameter for torch.stft
            window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        # Ensure window is on the same device as magnitude/phase
        self.window = self.window.to(magnitude.device)
        
        # Reconstruct the complex tensor from magnitude and phase
        complex_spec = magnitude * torch.exp(phase * 1j)

        inverse_transform = torch.istft(
            complex_spec,
            n_fft=self.filter_length, # n_fft parameter for torch.istft
            hop_length=self.hop_length,
            win_length=self.win_length, # win_length parameter for torch.istft
            window=self.window)

        # The original STFT class output had an extra dimension, 
        # so unsqueeze to maintain compatibility if ISTFTNetFE expects it.
        # (batch_size, num_samples) -> (batch_size, 1, num_samples)
        return inverse_transform.unsqueeze(-2)

    def forward(self, input_data):
        # Ensure window is on the same device as input_data
        self.window = self.window.to(input_data.device)
        
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction
