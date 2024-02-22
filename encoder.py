"""
Encoder for the VAE architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.loudness import LoudnessExtractor
from pitch_encoder.pitch import PitchEncoder

class MonoTimbreEncoder(nn.Module):
    def __init__(self, pitch_dim, spectrogram_dim, hidden_dim, z_dim):
        super(MonoTimbreEncoder, self).__init__()
        self.pitch_encoder = Encoder(pitch_dim, hidden_dim, z_dim)
        self.spectrogram_encoder = Encoder(spectrogram_dim, hidden_dim, z_dim)
    
    def forward(self, pitch, spectrogram):
        pitch_z = self.pitch_encoder(pitch)
        spectrogram_z = self.spectrogram_encoder(spectrogram)
        return pitch_z, spectrogram_z
    
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
                 device: str = 'mps'):
        
        super(Encoder, self).__init__()

        self.sr = sr
        self.frame_length = frame_length
        self.device = device

        self.loudness_extractor = LoudnessExtractor(sr, frame_length, device = device)

        if use_z:
            self.timbre_encoder = MonoTimbreEncoder(sr, 
                                                    n_fft, 
                                                    frame_length, 
                                                    n_mels, 
                                                    n_mfcc, 
                                                    gru_units, 
                                                    z_units, 
                                                    bidirectional)
            
    
    def forward(self, x):
        features = {}
        features['loudness'] = self.loudness_extractor(x)
        features['pitches'] = self.pitches_extractor(x)
        if self.use_z:
            features['timbre'] = self.timbre_encoder(x)
        return features
    
