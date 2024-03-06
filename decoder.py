"""
Decoder for the VAE architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

class MLP(nn.Module):
    """
    Multi-layer perceptron 

    Each layer consists of Dense -> LayerNorm -> ReLU 

    Args:
        input_dim: input dimension
        hidden_dims: hidden dimension
        layer_num: number of MLP Layers
        relu: ReLU, LeakyReLU, or PReLU etc.

    Input: Input tensor of size (batch, ... input_dim)
    Output: Output tensor of size (batch, ... hidden_dims)
    """

    def __init__(self, input_dim: int, 
                 hidden_dims: int, 
                 layer_num: int, 
                 relu: str = 'ReLU'):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layer_num = layer_num
        self.relu = getattr(nn, relu)()

        layers = []
        for i in range(self.layer_num):
            layers.append(nn.Linear(self.hidden_dims, self.hidden_dims))
            layers.append(nn.LayerNorm(self.hidden_dims))
            layers.append(self.relu)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
class Decoder(nn.Module):
    """
    Decoder, taking in dictionary of audio features and returning synthesiser parameters

    Args:
        use_z: whether to include timbre encoding
        mlp_hidden_dims: hidden dimension of mlp
        mlp_layer_num: number of mlp layers
        z_units: number of units in timbre encoding
        n_harmonics: number of harmonics in synthesiser
        n_freqs: number of frequency bins in synthesiser
        gru_units: number of units in gru layer
        bidirectional: whether to use bidirectional gru
    
    Input: Dictionary of audio features (pitches, amplitude, loudness, timbre)
        pitches: Pitch features of size (batch, voices, frames)
        amplitude: Amplitude features of size (batch, voices, frames)
        loudness: Loudness features of size (batch, frames)
        timbre: Timbre features of size (batch, z_units, frames)

    Output: Dictionary of synthesiser parameters (pitches, harmonics, amplitude, noise)
        frequencies: Frequency features of size (batch, voices, frames)
        harmonics: Harmonics spectra (batch, voices, n_harmonics, frames)
        amplitude: Amplitude envelope (batch, voices, frames)
        noise: Noise filter coefficients of size (batch, filter_coeff, frames)
    """
    
    def __init__(self, 
                 use_z: bool = True,
                 mlp_hidden_dims: int = 512,
                 mlp_layer_num: int = 3,
                 z_units: int = 16,
                 n_harmonics: int = 101,
                 n_freqs: int = 65,
                 gru_units: int = 512,
                 bidirectional: bool = True,
                 max_voices: int = 10):
        
        super(Decoder, self).__init__()

        self.use_z = use_z
        self.mlp_hidden_dims = mlp_hidden_dims
        self.mlp_layer_num = mlp_layer_num
        self.z_units = z_units
        self.n_harmonics = n_harmonics
        self.n_freqs = n_freqs
        self.gru_units = gru_units
        self.bidirectional = bidirectional
        self.max_voices = max_voices
        

        # Pitch pipeline 
        self.pitch_mlp_f0 = MLP(input_dim=1, hidden_dims=mlp_hidden_dims, layer_num=mlp_layer_num)
        self.pitch_mlp_loudness = MLP(input_dim=1, hidden_dims=mlp_hidden_dims, layer_num=mlp_layer_num)
        self.pitch_mlp_velocity = MLP(input_dim=1, hidden_dims=mlp_hidden_dims, layer_num=mlp_layer_num)

        # Noise pipeline
        self.noise_mlp_f0 = MLP(input_dim=1, hidden_dims=mlp_hidden_dims, layer_num=mlp_layer_num)
        self.noise_mlp_velocity = MLP(input_dim=1, hidden_dims=mlp_hidden_dims, layer_num=mlp_layer_num)

        # Timbre pipeline
        if self.use_z:
            self.timbre_mlp = MLP(input_dim=z_units, hidden_dims=mlp_hidden_dims, layer_num=mlp_layer_num)
            num_mlp = 4
        else:  
            num_mlp = 3
        
        self.pitch_gru = nn.GRU(input_size=mlp_hidden_dims * num_mlp, 
                                hidden_size=gru_units, 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=bidirectional)
        
        self.pitch_mlp = MLP(input_dim=gru_units * 2 if bidirectional else gru_units,
                             hidden_dims=mlp_hidden_dims, 
                             layer_num=mlp_layer_num)
        
        self.noise_gru = nn.GRU(input_size=mlp_hidden_dims * 2, 
                                hidden_size=gru_units, 
                                num_layers=1, 
                                batch_first=True, 
                                bidirectional=bidirectional)
        
        self.noise_mlp = MLP(input_dim=gru_units * 2 if bidirectional else gru_units,
                             hidden_dims=mlp_hidden_dims, 
                             layer_num=mlp_layer_num)
        
        self.dense_harmonic = nn.Linear(mlp_hidden_dims, n_harmonics)
        self.dense_filter = nn.Linear(mlp_hidden_dims, n_freqs + 1)        

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError