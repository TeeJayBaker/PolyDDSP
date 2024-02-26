"""
Encoder for the VAE architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

from modules.loudness import LoudnessExtractor
from pitch_encoder.pitch import PitchEncoder

class MonoTimbreEncoder(nn.Module):
    def __init__(self, sr: int = 16000,
                 frame_length: int = 64,
                 use_z: bool = True,
                 z_units: int = 16,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 n_mfcc: int = 30,
                 gru_units: int = 512,
                 bidirectional: bool = True):
        super(MonoTimbreEncoder, self).__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft, hop_length=frame_length, n_mels=n_mels, f_min=20.0, f_max=8000.0,
            ),
        )

        self.norm = nn.InstanceNorm1d(n_mfcc, affine=True)
        self.permute = lambda x: x.permute(0, 2, 1)
        self.gru = nn.GRU(
            input_size=n_mfcc,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dense = nn.Linear(gru_units * 2 if bidirectional else gru_units, z_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mfcc(x)
        x = x[:, :, :-1]
        x = self.norm(x)
        x = self.permute(x)
        x, _ = self.gru(x)
        x = self.dense(x)
        return x
    
class Encoder(nn.Module):
    """
    Encoder, taking in raw audio and returning audio features (pitches, amplitude, loudness, timbre)

    Args:
        sr: input audio sample rate
        use_z: whether to use timbre encoding
        
            z_units: number of units in timbre encoding
            n_fft: number of fft bins
            n_mels: number of mel bins
            n_mfcc: number of mfcc bins
            gru_units: number of units in gru layer
            bidirectional: whether to use bidirectional gru
        
        device: Specify whether computed on cpu, cuda or mps
    
    Input: Audio input of size (batch, samples)
    Output: Dictionary of audio features (pitches, amplitude, loudness, timbre)
        pitches: Pitch features of size (batch, voices, frames
        amplitude: Amplitude features of size (batch, voices, frames)
        loudness: Loudness features of size (batch, frames)
        timbre: Timbre features of size (batch, frames, z_units)
    """
    def __init__(self, sr: int = 16000,
                 frame_length: int = 64,
                 use_z: bool = True,
                 z_units: int = 16,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 n_mfcc: int = 30,
                 gru_units: int = 512,
                 bidirectional: bool = True,
                 device: str = 'cpu'):
        
        super(Encoder, self).__init__()

        self.sr = sr
        self.frame_length = frame_length
        self.device = device

        self.loudness_extractor = LoudnessExtractor(sr, frame_length, device = device)
        self.pitch_encoder = PitchEncoder(device = device)
        self.use_z = use_z

        if self.use_z:
            self.timbre_encoder = MonoTimbreEncoder(sr, 
                                                    n_fft, 
                                                    frame_length, 
                                                    n_mels, 
                                                    n_mfcc, 
                                                    gru_units, 
                                                    z_units, 
                                                    bidirectional)
            
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        features['loudness'] = self.loudness_extractor(x)
        features['pitches'] = self.pitch_encoder(x)
        if self.use_z:
            features['timbre'] = self.timbre_encoder(x)
        return features
    
import librosa
audio = librosa.load("pitch_encoder/01_BN2-131-B_solo_mic.wav", sr=22050)[0]
audio = torch.tensor(audio).unsqueeze(0).to('cpu')
model = Encoder()
output = model(audio)
print(output)