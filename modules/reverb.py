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
        device: Specify whether computed on cpu, cuda or mps

    Input: Audio input of size (batch, samples)
    Output: Reverberated audio of size (batch, samples)
    """
    def __init__(self,
                trainable = True,
                reverb_length = 16000,
                add_dry = True,
                device = 'mps'):
        
        super(Reverb, self).__init__()

        self.trainable = trainable
        self.reverb_length = reverb_length
        self.add_dry = add_dry

        if self.trainable:
            self.impulse_response = nn.Parameter(torch.zeros(self.reverb_length), 
                                                 requires_grad = True)
        else:
            self.impulse_response = nn.Parameter(torch.zeros(self.reverb_length), 
                                                 requires_grad = False)
            
    def forward(self, audio):
        raise NotImplementedError