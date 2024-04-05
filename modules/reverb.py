"""
Function for Finite Impulse Response (FIR) reverb
"""

import torch
import torch.nn as nn
import modules.operations as ops


class Reverb(nn.Module):
    """
    Finite Impulse Response (FIR) reverb module

    Args:
        trainable: Learn the impulse_response as a single variable for the entire
            dataset.
        reverb_length: Length of the impulse response. Only used if
            trainable=True.
        add_dry: Add dry signal to reverberated signal on output.
        impulse_response: Impulse response to use if trainable=False.
        device: Specify whether computed on cpu, cuda or mps

    Input: Audio input of size (batch, samples)
    Output: Reverberated audio of size (batch, samples)
    """

    def __init__(
        self,
        trainable: bool = True,
        reverb_length: int = 16000,
        add_dry: bool = True,
        impulse_response: bool = None,
        device: str = "mps",
    ):
        super(Reverb, self).__init__()

        self.trainable = trainable
        self.reverb_length = reverb_length
        self.add_dry = add_dry

        if self.trainable:
            self.impulse_response = nn.Parameter(
                torch.zeros(self.reverb_length), requires_grad=True
            )
        else:
            if impulse_response is None:
                raise ValueError('Must provide "ir" tensor if Reverb trainable=False.')
            self.impulse_response = impulse_response

        self.to(device)

    def _mask_dry(self, impulse_response: torch.Tensor) -> torch.Tensor:
        """
        set first ir to zero to mask dry signal from reverberated signal
        """
        # Add batch if 1d
        if len(impulse_response.shape) == 1:
            impulse_response = impulse_response.unsqueeze(0)
        # delete channel dimensions if 3d
        if len(impulse_response.shape) == 3:
            impulse_response = impulse_response.squeeze(1)

        dry_mask = torch.zeros((int(impulse_response.shape[0]), 1))
        return torch.cat([dry_mask, impulse_response[:, 1:]], dim=1)

    def _match_dimensions(
        self, audio: torch.Tensor, impulse_response: torch.Tensor
    ) -> torch.Tensor:
        """
        Match dimensions of audio and impulse response via batch size
        """
        # Add batch if 1d
        if len(impulse_response.shape) == 1:
            impulse_response = impulse_response[None, :]
        # tile to match batch dim
        batch_size = int(audio.shape[0])
        return torch.tile(impulse_response, [batch_size, 1])

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply impulse response to audio
        """
        audio = torch.FloatTensor(audio)
        impulse_response = torch.FloatTensor(self.impulse_response)

        if self.trainable:
            impulse_response = self._match_dimensions(audio, impulse_response)
        else:
            if self.impulse_response is None:
                raise ValueError('Must provide "ir" tensor if Reverb trainable=False.')

        impulse_response = self._mask_dry(impulse_response)
        wet = ops.fft_convolve(
            audio, impulse_response, padding="same", delay_compensation=0
        )
        return (wet + audio) if self.add_dry else wet
