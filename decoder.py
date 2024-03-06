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