"""
Decoder for the AE architecture
"""

import torch
import torch.nn as nn


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

    def __init__(
        self,
        input_dim: int,
        hidden_dims: int,
        layer_num: int,
        relu: str = "ReLU",
        device: str = "cpu",
    ):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.layer_num = layer_num
        self.relu = getattr(nn, relu)()

        layers = []
        if layer_num > 0:
            layers.append(nn.Linear(input_dim, hidden_dims))
            layers.append(nn.LayerNorm(hidden_dims))
            layers.append(self.relu)
        if layer_num > 1:
            for i in range(layer_num - 1):
                layers.append(nn.Linear(hidden_dims, hidden_dims))
                layers.append(nn.LayerNorm(hidden_dims))
                layers.append(self.relu)

        self.mlp = nn.Sequential(*layers)

        self.to(device)

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
        harmonics: Harmonics spectra (batch, voices, harmonics, frames)
        amplitude: per voice amplitude envelope (batch, voices, frames)
        noise: Noise filter coefficients of size (batch, filter_coeff, frames)
    """

    def __init__(
        self,
        use_z: bool = True,
        mlp_hidden_dims: int = 512,
        mlp_layer_num: int = 3,
        z_units: int = 16,
        n_harmonics: int = 101,
        n_freqs: int = 65,
        gru_units: int = 512,
        bidirectional: bool = True,
        max_voices: int = 10,
        device: str = "cpu",
    ):
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

        self.f0_mlp = MLP(
            input_dim=1,
            hidden_dims=mlp_hidden_dims,
            layer_num=mlp_layer_num,
            device=device,
        )
        self.amplitude_mlp = MLP(
            input_dim=1,
            hidden_dims=mlp_hidden_dims,
            layer_num=mlp_layer_num,
            device=device,
        )
        self.loudness_mlp = MLP(
            input_dim=1,
            hidden_dims=mlp_hidden_dims,
            layer_num=mlp_layer_num,
            device=device,
        )

        # Timbre pipeline
        if self.use_z:
            self.timbre_mlp = MLP(
                input_dim=z_units,
                hidden_dims=mlp_hidden_dims,
                layer_num=mlp_layer_num,
                device=device,
            )
            num_mlp = 4
        else:
            num_mlp = 3

        self.decoder_gru = nn.GRU(
            input_size=mlp_hidden_dims * num_mlp,
            hidden_size=gru_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            device=device,
        )

        self.decoder_mlp = MLP(
            input_dim=gru_units * 2 if bidirectional else gru_units,
            hidden_dims=mlp_hidden_dims,
            layer_num=mlp_layer_num,
            device=device,
        )

        self.dense_harmonic = nn.Linear(mlp_hidden_dims, n_harmonics + 1, device=device)
        self.dense_filter = nn.Linear(mlp_hidden_dims, n_freqs, device=device)

        self.to(device)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pitch = x["pitch"].unsqueeze(-1)
        amplitude = x["amplitude"].unsqueeze(-1)
        loudness = x["loudness"].unsqueeze(-1)
        timbre = x["timbre"]

        batch = pitch.shape[0]

        pitch = self.f0_mlp(pitch).reshape(
            batch * self.max_voices, -1, self.mlp_hidden_dims
        )
        amplitude = self.amplitude_mlp(amplitude).reshape(
            batch * self.max_voices, -1, self.mlp_hidden_dims
        )
        loudness = self.loudness_mlp(loudness).repeat(self.max_voices, 1, 1)

        if self.use_z:
            timbre = timbre.permute(0, 2, 1)
            timbre = self.timbre_mlp(timbre)
            timbre = timbre.repeat(self.max_voices, 1, 1)

            latent = torch.cat([pitch, amplitude, loudness, timbre], dim=-1)
        else:
            latent = torch.cat([pitch, amplitude, loudness], dim=-1)

        latent = self.decoder_gru(latent)[0]
        latent = self.decoder_mlp(latent)

        # need to reconstruct polyphony
        latent = latent.reshape(batch, self.max_voices, -1, self.mlp_hidden_dims)
        harm_amp = self.dense_harmonic(latent)

        harm_out = harm_amp[..., 1:].softmax(dim=-1).permute(0, 1, 3, 2)
        amp_out = modified_sigmoid(harm_amp[..., 0])

        noise = self.dense_filter(latent).softmax(dim=-1).permute(0, 1, 3, 2)

        return {
            "pitch": x["pitch"],
            "harmonics": harm_out,
            "amplitude": amp_out,
            "noise": noise,
        }


def modified_sigmoid(a):
    a = a.sigmoid()
    a = a.pow(2.3026)  # log10
    a = a.mul(2.0)
    a.add_(1e-7)
    return a
