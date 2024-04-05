"""
timbre encoder, that takes in pitch and spectrogram
and returns a source separated spectrogram and MFCCs
"""

import torch
import torchaudio
import torch.nn as nn


class TimbreEncoder(nn.module):
    def __init__(self):
        super(TimbreEncoder, self).__init__()
        self.pitch_encoder = nn.Sequential(nn.Linear(1))
