"""
Autoencoder for the AE architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

from decoder import Decoder
from encoder import Encoder
from modules.reverb import Reverb
from modules.noise import FilteredNoise
from modules.synth import AdditiveSynth


class AutoEncoder(nn.Module):
    """
    Autoencoder for the AE architecture
    """

    def __init__(self, 
                 use_z: bool = False, 
                 mlp_hidden_dims: int = 512,
                 reverb_length: int = 16000,
                 add_dry: bool = True,
                 impulse_response: bool = None,
                 device: str = 'mps'):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(use_z=use_z, mlp_hidden_dims=mlp_hidden_dims)
        self.reverb = Reverb(reverb_length=reverb_length, add_dry=add_dry, impulse_response=impulse_response, device=device)
        self.noise = FilteredNoise()
        self.synth = AdditiveSynth()

    def forward(self, x):
        parts = self.encoder(x)
        output = self.decoder(parts)
        audio = self.synth(output)
        noise = self.noise(output['noise'])
        reverbed = self.reverb(audio + noise)
        audio = audio[:, :x.shape[1]]
        noise = noise[:, :x.shape[1]]
        reverbed = reverbed[:, :x.shape[1]]
        return audio, noise, reverbed
    

import librosa
audio = librosa.load("pitch_encoder/01_BN2-131-B_solo_mic.wav", sr=22050, duration=10)[0]

print(audio.shape)
audio = torch.tensor(audio).unsqueeze(0).to('cpu')

model = AutoEncoder()
audio, noise, reverbed = model(audio)
print(audio.shape, noise.shape, reverbed.shape)