"""Single shared decoder, joint (amp, harm_dist, noise_mags) head per voice.

Generalises DDSP's `ddsp.decoders.RnnFcDecoder` to V voices via batch fold. One
GRU's hidden state drives all three heads — that's the load-bearing structural
property: it forces gradient coupling between the amplitude, harmonic
distribution and filtered-noise magnitudes so the optimiser can't drain one head
while another collapses (the "IR-as-amplifier" trap that PolyDDSP's earlier
split-decoder design exhibited).

Voice fold is `(B,V,T,...) → (B*V, T, ...)` so the same weights process every
voice slot. Voice identity is unstable (FIFO with eviction in `pitch.py`), so V
independent decoders would have to relearn voice-permutation invariance — wasteful.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _modified_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sigmoid(x).pow(math.log(10.0)) + 1e-7


def _glorot_dense(in_dim: int, out_dim: int) -> nn.Linear:
    """`nn.Linear` initialised to match TF Keras `Dense` defaults:
    `kernel_initializer='glorot_uniform'`, `bias_initializer='zeros'`.

    PyTorch's `nn.Linear` defaults are `kaiming_uniform(a=sqrt(5))` for the
    weight and a non-zero uniform for the bias (`U(-1/sqrt(in), +1/sqrt(in))`).
    The non-zero bias on the joint head's amp slot (the first output channel)
    can land at a small-but-significant negative value at init, which makes
    `_modified_sigmoid(amp_logit)` start far below 1.0 and forces the optimiser
    to climb back. DDSP doesn't have this trap because its bias is exactly 0.
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _MLP(nn.Module):
    """Stack of (Linear → LayerNorm → LeakyReLU(0.2)) layers.

    Matches DDSP's `ddsp.nn.Fc`/`ddsp.nn.FcStack`: TF Keras `Dense` defaults to
    glorot_uniform weights + zero bias, `LayerNormalization` defaults to
    `epsilon=1e-3`, and `tf.nn.leaky_relu` defaults to `alpha=0.2`.
    """

    def __init__(self, in_dim: int, hidden: int = 512, layers: int = 3) -> None:
        super().__init__()
        seq: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            seq += [_glorot_dense(d, hidden), nn.LayerNorm(hidden, eps=1e-3), nn.LeakyReLU(0.2)]
            d = hidden
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _stack_voice_features(
    pitch: torch.Tensor,
    velocity: torch.Tensor | None,
    loudness: torch.Tensor,
    z: torch.Tensor | None,
) -> torch.Tensor:
    """Fold voice slots into the batch dim. Returns `(B*V, T, in_dim)`.

    `velocity` is omitted when `None` (solo case: it's identically 1.0 and an
    MLP over a constant stream just wastes ~530K params).
    """
    B, V, T = pitch.shape
    pitch_flat = pitch.reshape(B * V, T, 1)
    loud_rep = loudness.unsqueeze(1).expand(B, V, T).reshape(B * V, T, 1)
    parts = [pitch_flat]
    if velocity is not None:
        parts.append(velocity.reshape(B * V, T, 1))
    parts.append(loud_rep)
    if z is not None:
        z_dim = z.shape[-1]
        z_rep = z.unsqueeze(1).expand(B, V, T, z_dim).reshape(B * V, T, z_dim)
        parts.append(z_rep)
    return torch.cat(parts, dim=-1)


class MonoDecoder(nn.Module):
    """Per-voice decoder with joint (amp, harm_dist, noise_mags) head.

    DDSP's `RnnFcDecoder` generalised to V voices via batch fold. Inputs are
    `(B, V, T, ...)`; the decoder folds to `(B*V, T, ...)`, runs one shared
    GRU + post-MLP, and emits a single Dense head whose output is split into:

        amp_v:      (B, V, T)                  — modified-sigmoid bounded
        harm_dist:  (B, V, T, n_harmonics)     — raw logits
        noise_mags: (B, V, T, n_bands)         — raw logits

    The synthesisers (`AdditiveSynth`, `FilteredNoise`) apply their own
    activations (exp_sigmoid, normalize_harmonics, etc.) — DDSP convention.
    """

    def __init__(
        self,
        z_dim: int = 16,
        n_harmonics: int = 60,
        n_bands: int = 65,
        gru_hidden: int = 512,
        mlp_hidden: int = 512,
        mlp_layers: int = 3,
        use_z: bool = False,
        use_velocity: bool = True,
    ) -> None:
        super().__init__()
        self.use_z = use_z
        self.use_velocity = use_velocity
        self.n_harmonics = n_harmonics
        self.n_bands = n_bands

        self.f0_mlp = _MLP(1, mlp_hidden, mlp_layers)
        if use_velocity:
            self.vel_mlp = _MLP(1, mlp_hidden, mlp_layers)
        self.loud_mlp = _MLP(1, mlp_hidden, mlp_layers)
        if use_z:
            self.z_mlp = _MLP(z_dim, mlp_hidden, mlp_layers)
        n_streams = 2 + int(use_velocity) + int(use_z)
        cat_dim = n_streams * mlp_hidden

        self.gru = nn.GRU(cat_dim, gru_hidden, batch_first=True)
        for name, p in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # The post-MLP sees the GRU output concatenated with its own input
        # (residual skip), as in DDSP's `RnnFcDecoder`.
        self.post_mlp = _MLP(gru_hidden + cat_dim, mlp_hidden, mlp_layers)
        # Joint head: 1 (amp) + n_harmonics (harm_dist) + n_bands (noise_mags).
        # DDSP's `tfkl.Dense(n_out)` defaults to glorot_uniform + zero bias;
        # PyTorch's `nn.Linear` bias defaults to a non-zero uniform, which lands
        # the amp-slot bias at a small but significant negative value on init.
        self.head = _glorot_dense(mlp_hidden, 1 + n_harmonics + n_bands)

    def forward(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        loudness: torch.Tensor,
        z: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, V, T = pitch.shape
        vel_in = velocity if self.use_velocity else None
        feat_in = _stack_voice_features(pitch, vel_in, loudness, z if self.use_z else None)

        idx = 0
        feats = [self.f0_mlp(feat_in[..., idx:idx + 1])]
        idx += 1
        if self.use_velocity:
            feats.append(self.vel_mlp(feat_in[..., idx:idx + 1]))
            idx += 1
        feats.append(self.loud_mlp(feat_in[..., idx:idx + 1]))
        idx += 1
        if self.use_z:
            feats.append(self.z_mlp(feat_in[..., idx:]))
        cat = torch.cat(feats, dim=-1)

        gru_out, _ = self.gru(cat)
        post = self.post_mlp(torch.cat([cat, gru_out], dim=-1))
        head_out = self.head(post)  # (B*V, T, 1+H+n_bands)

        amp_v = _modified_sigmoid(head_out[..., 0]).reshape(B, V, T)
        harm_dist = head_out[..., 1:1 + self.n_harmonics].reshape(B, V, T, self.n_harmonics)
        noise_mags = head_out[..., 1 + self.n_harmonics:].reshape(B, V, T, self.n_bands)
        return harm_dist, amp_v, noise_mags
