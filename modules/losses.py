"""
Loss functions for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SpectralLoss(nn.Module):
    """
    Multi-scale spectral loss function
    """
    def __init__(self,
                fft_sizes = [2048, 1024, 512, 256, 128, 64],
                epsilon = 1e-7,
                overlap = 0.75,
                loss_type = 'L1',
                mag_weight = 1.0,
                logmag_weight = 1.0,
                device = 'mps'):
        
        super(SpectralLoss, self).__init__()

        self.fft_sizes = fft_sizes
        self.epsilon = epsilon
        self.overlap = overlap
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight
        self.device = device

        if loss_type == 'L1':
            self.loss = F.l1_loss
        elif loss_type == 'L2':
            self.loss = F.mse_loss
        elif loss_type == 'cosine':
            self.loss = F.cosine_similarity
        else:
            raise ValueError('loss_type must be one of "L1", "L2", or "cosine"')
        
    def spectogram(self, audio: torch.Tensor, 
                   fft_size: list, 
                   power: int = 1):
        """
        Compute spectogram from audio

        Args:
            audio: input audio tensor
            fft_size: size of fft window
            power: power of spectogram

        Returns: spectogram of audio
        """
        audio = torch.FloatTensor(audio)
        audio = audio.to(self.device)
        return torchaudio.transforms.Spectrogram(n_fft = fft_size, power = power)(audio)

    def forward(self, x, y):
        """
        Compute multi-scale spectral loss

        Args:
            x: input audio tensor
            y: target audio tensor
        
        Returns: multi-scale spectral loss
        """
        loss = 0
        for fft_size in self.fft_sizes:
            x_mag = self.spectogram(x, fft_size).to(self.device)
            y_mag = self.spectogram(y, fft_size).to(self.device)
            loss += self.loss(x_mag, y_mag) * self.mag_weight

            x_logmag = torch.log(x_mag + self.epsilon)
            y_logmag = torch.log(y_mag + self.epsilon)
            loss += self.loss(x_logmag, y_logmag) * self.logmag_weight

        return loss