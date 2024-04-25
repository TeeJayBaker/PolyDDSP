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

    Args:
        fft_sizes: list of fft sizes to compute loss over
        epsilon: small value to avoid log(0)
        overlap: overlap between frames
        loss_type: loss function to use
        mag_weight: weight for magnitude loss
        logmag_weight: weight for log magnitude loss
        device: Specify whether computed on cpu, cuda or mps
            (bearing in mind fft is not implemented on mps)
    """

    def __init__(
        self,
        fft_sizes: list[int] = [2048, 1024, 512, 256, 128, 64],
        epsilon: float = 1e-7,
        overlap: float = 0.75,
        loss_type: str = "L1",
        mag_weight: float = 1.0,
        logmag_weight: float = 1.0,
        device: str = "cpu",
    ):
        super(SpectralLoss, self).__init__()

        self.fft_sizes = fft_sizes
        self.epsilon = epsilon
        self.overlap = overlap
        self.mag_weight = mag_weight
        self.logmag_weight = logmag_weight
        self.device = device

        if loss_type == "L1":
            self.loss = F.l1_loss
        elif loss_type == "L2":
            self.loss = F.mse_loss
        elif loss_type == "cosine":
            self.loss = F.cosine_similarity
        else:
            raise ValueError('loss_type must be one of "L1", "L2", or "cosine"')

    def spectogram(self, audio: torch.Tensor, fft_size: list[int], power: int = 1):
        """
        Compute spectogram from audio

        Args:
            audio: input audio tensor
            fft_size: size of fft window
            power: power of spectogram

        Returns: spectogram of audio
        """
        return torchaudio.transforms.Spectrogram(
            n_fft=fft_size, power=power, wkwargs={"device": audio.device}
        )(audio)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute multi-scale spectral loss

        Args:
            x: input audio tensor
            y: target audio tensor

        Returns: multi-scale spectral loss
        """
        for fft_size in self.fft_sizes:
            x_mag = self.spectogram(x, fft_size)
            y_mag = self.spectogram(y, fft_size)
            mag_loss = self.loss(x_mag, y_mag) * self.mag_weight

            x_logmag = torch.log(x_mag + self.epsilon)
            y_logmag = torch.log(y_mag + self.epsilon)
            log_loss = self.loss(x_logmag, y_logmag) * self.logmag_weight

        return mag_loss + log_loss
