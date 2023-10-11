"""
Filtered noise synthesiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FilteredNoise(nn.Module):
    """
    Filtered noise synthesiser

    Args:
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        device: Specify whether computed on cpu, cuda or mps

    Input: Filter coefficients of size (batch, frames, banks)
    Output: Filtered noise audio (batch, samples)
    """
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2, device = 'mps'):
        super(FilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain

    def forward(self, z):
        """
        Compute LTI-FVR filter banks, and calculate time varying filtered noise via overlap-add.
        """

        batch_size, num_frames, num_banks = z['H'].shape