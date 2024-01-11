"""
Function for Finite Impulse Response (FIR) reverb
"""

import torch
import torch.nn as nn

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
    def __init__(self,
                trainable = True,
                reverb_length = 16000,
                add_dry = True,
                impulse_response = None,
                device = 'mps'):
        
        super(Reverb, self).__init__()

        self.trainable = trainable
        self.reverb_length = reverb_length
        self.add_dry = add_dry

        if self.trainable:
            self.impulse_response = nn.Parameter(torch.zeros(self.reverb_length), 
                                                 requires_grad = True)
        else:
            if impulse_response is None:
                raise ValueError('Must provide "ir" tensor if Reverb trainable=False.')
            self.impulse_response = impulse_response

    def _mask_dry(self, impulse_response):
        """
        set first ir to zero to mask dry signal from reverberated signal
        """
        # Add batch if 1d
        if len(impulse_response.shape) == 1:
            impulse_response = impulse_response.unsqueeze(0)
        # delete channel dimensions if 3d
        if len(impulse_response.shape) == 3:
            impulse_response = impulse_response.squeeze(1)

        dry_mask = torch.zeros([int(impulse_response.shape[0]), 1], dtype = torch.FloatTensor)
        return torch.cat([dry_mask, impulse_response[:, 1:]], dim=1)
    
    def _match_dimensions(self, audio, impulse_response):
        """
        Match dimensions of audio and impulse response via batch size
        """
        raise NotImplementedError
            
    def forward(self, audio):
        """
        Apply impulse response to audio
        """
        audio = torch.FloatTensor(audio)
        self.impulse_response = torch.FloatTensor(self.impulse_response)
        self.impulse_response = self.impulse_response.to(self.device)

        raise NotImplementedError