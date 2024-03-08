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
    