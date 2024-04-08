"""
Encoder for the AE architecture
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np

from modules.loudness import LoudnessExtractor
from pitch_encoder.pitch import PitchEncoder


class MonoTimbreEncoder(nn.Module):
    """
    Encoder for timbre features

    Args:
        sr: input audio sample rate
        hop_length: length of each frame shift
        z_units: number of units in timbre encoding
        n_fft: number of fft bins
        n_mels: number of mel bins
        n_mfcc: number of mfcc bins
        gru_units: number of units in gru layer
        bidirectional: whether to use bidirectional gru

    Input: Audio input of size (batch, samples)
    Output: Timbre features of size (batch, frames, z_units)
    """

    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 64,
        z_units: int = 16,
        n_fft: int = 2048,
        n_mels: int = 128,
        n_mfcc: int = 30,
        gru_units: int = 512,
        bidirectional: bool = True,
        device: str = "cpu",
    ):
        super(MonoTimbreEncoder, self).__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=20.0,
                f_max=8000.0,
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

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mfcc(x)
        x = self.norm(x)
        x = self.permute(x)
        x, _ = self.gru(x)
        x = self.dense(x)
        x = self.permute(x)
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

    def __init__(
        self,
        sr: int = 22050,
        frame_length: int = 64,
        use_z: bool = True,
        z_units: int = 16,
        n_fft: int = 2048,
        n_mels: int = 128,
        n_mfcc: int = 30,
        gru_units: int = 512,
        bidirectional: bool = True,
        device: str = "cpu",
    ):
        super(Encoder, self).__init__()

        self.sr = sr
        self.frame_length = frame_length
        self.use_z = use_z

        self.loudness_extractor = LoudnessExtractor(sr, frame_length, device=device)
        self.pitch_encoder = PitchEncoder(device=device)

        if self.use_z:
            self.timbre_encoder = MonoTimbreEncoder(
                sr,
                frame_length,
                z_units,
                n_fft,
                n_mels,
                n_mfcc,
                gru_units,
                bidirectional,
            )

        self.to(device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        num_frames = int(np.ceil(x.shape[-1] / self.frame_length))
        upsample = nn.Upsample(size=(num_frames), mode="linear")
        features = {}
        features["loudness"] = self.loudness_extractor(x)
        features["pitch"], features["amplitude"] = self.pitch_encoder(x)
        features["pitch"] = upsample(features["pitch"])
        features["amplitude"] = upsample(features["amplitude"])
        if self.use_z:
            features["timbre"] = self.timbre_encoder(x)
        return features


# import librosa

# audio = librosa.load("pitch_encoder/01_BN2-131-B_solo_mic.wav", sr=22050, duration=10)[
#     0
# ]

# print(audio.shape)
# audio = torch.tensor(audio).unsqueeze(0).to("cpu")
# model = Encoder()
# output = model(audio)
# print("Pitch: ", output["pitch"].shape)
# print("Amplitude: ", output["amplitude"].shape)
# print("Loudness: ", output["loudness"].shape)
# print("Timbre: ", output["timbre"].shape)

# # print a plot of pitch and loudness on the same x axis
# import matplotlib.pyplot as plt
# import numpy as np

# plt.figure(figsize=(10, 6))
# # log scale pitch from frequency to midi
# plt.plot(
#     np.arange(output["pitch"].shape[-1]),
#     12 * np.log2(output["pitch"].squeeze().numpy().T),
# )
# # loudness
# plt.plot(
#     np.arange(output["loudness"].shape[-1]), output["loudness"].squeeze().numpy() + 200
# )
# plt.legend(["Pitch", "Loudness"])

# plt.show()
