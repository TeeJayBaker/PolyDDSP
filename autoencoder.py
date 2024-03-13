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
    

import librosa
audio = librosa.load("pitch_encoder/01_BN2-131-B_solo_mic.wav", sr=22050, duration=10)[0]

print(audio.shape)
audio = torch.tensor(audio).unsqueeze(0).to('cpu')
encoder = Encoder()
decoder = Decoder()
parts = encoder(audio)
output = decoder(parts)
print(output['pitch'].shape)
print(output['amplitude'].shape)
print(output['harmonics'].shape)
print(output['noise'].shape)
synth = AdditiveSynth()
audio = synth(output)
print(audio.shape)

