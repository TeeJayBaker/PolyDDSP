"""
Pitch prediction module, porting basic_pitch to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class basic_pitch(nn.Module):
    """
    Port of basic_pitch pitch prediction to pytorch
    """

    def __init__(self,
                sr=16000,
                n_fft=2048,
                hop_length=512,
                n_mels=256,
                fmin=0,
                fmax=None,
                device='mps'):
        super(basic_pitch, self).__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.device = device

        self.mel_basis = nn.Parameter(torch.tensor(
            torchaudio.functional.create_fb_matrix(
                n_freqs=self.n_fft // 2 + 1,
                f_min=self.fmin,
                f_max=self.fmax,
                n_mels=self.n_mels,
                sample_rate=self.sr,
                norm='slaney',
                mel_scale='htk',
                dtype=torch.float32,
                device=self.device)), requires_grad=False)
            
    def forward():
        raise NotImplementedError
        