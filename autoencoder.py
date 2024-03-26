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
                 sr: int = 22050, # General Variables
                 frame_length: int = 64,
                 use_z: bool = True, 
                 z_units: int = 16,
                 n_fft: int = 2048,
                 n_mels: int = 128,
                 n_mfcc: int = 30,
                 gru_units: int = 512,
                 bidirectional: bool = True,
                 mlp_hidden_dims: int = 512,
                 mlp_layer_num: int = 3,
                 n_harmonics: int = 101,
                 n_freqs: int = 65,
                 max_voices: int = 10,
                 trainable_reverb: bool = True, # Reverb Variables
                 reverb_length: int = 22050,
                 add_dry: bool = True,
                 impulse_response: bool = None,
                 normalize_below_nyquist: bool = True, # Synth Variables
                 amp_resample_method: str = 'window',
                 use_angular_cumsum: bool = False,  
                 attenuate_gain: float = 0.02,
                 initial_bias: float = -5.0, # Noise Variables
                 window_size: int = 257, 
                 device: str = 'cpu'):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(sr=sr,
                               frame_length=frame_length,
                               use_z=use_z,
                               z_units=z_units,
                               n_fft=n_fft,
                               n_mels=n_mels,
                               n_mfcc=n_mfcc,
                               gru_units=gru_units,
                               bidirectional=bidirectional,
                               device=device)
        
        self.decoder = Decoder(use_z=use_z, 
                               mlp_hidden_dims=mlp_hidden_dims,
                               mlp_layer_num=mlp_layer_num,
                               z_units=z_units,
                               n_harmonics=n_harmonics,
                               n_freqs=n_freqs,
                               gru_units=gru_units,
                               bidirectional=bidirectional,
                               max_voices=max_voices,
                               device=device)
        
        self.synth = AdditiveSynth(sample_rate=sr,
                                   normalize_below_nyquist=normalize_below_nyquist,
                                   amp_resample_method=amp_resample_method,
                                   use_angular_cumsum=use_angular_cumsum,
                                   frame_length=frame_length,
                                   attenuate_gain=attenuate_gain,
                                   device=device)
        
        self.noise = FilteredNoise(frame_length=frame_length,
                                   attenuate_gain=attenuate_gain,
                                   initial_bias=initial_bias,
                                   window_size=window_size,
                                   device=device)

        self.reverb = Reverb(trainable=trainable_reverb,
                                    reverb_length=reverb_length, 
                                    add_dry=add_dry, 
                                    impulse_response=impulse_response, 
                                    device=device)

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
    

# import librosa
# audio = librosa.load("pitch_encoder/01_BN2-131-B_solo_mic.wav", sr=22050, duration=10)[0]

# print(audio.shape)
# audio = torch.tensor(audio).unsqueeze(0).to('cpu')

# model = AutoEncoder()
# audio, noise, reverbed = model(audio)
# print(audio.shape, noise.shape, reverbed.shape)