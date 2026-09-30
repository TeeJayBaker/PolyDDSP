"""Frozen Basic Pitch (Spotify ICASSP 2022) + voice allocation.

The CNN architecture matches the bundled `bp_pytorch.pth` (151 KB) — same
shapes as Spotify's published TF model, ported once by
`scripts/port_basic_pitch.py`.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features.cqt import CQT2010v2

from polyddsp.model.note_extraction import (
    drop_overlapping_pitch_bends,
    get_pitch_bends,
    output_to_notes_polyphonic,
)
from polyddsp.model.voice_allocation import allocate_to_voices

WEIGHTS_PATH = Path(__file__).parent / "weights" / "bp_pytorch.pth"
BP_NATIVE_SR = 22_050
BP_FFT_HOP = 256
BP_OVERLAP_FRAMES = 30
AUDIO_N_SAMPLES = BP_NATIVE_SR * 2 - BP_FFT_HOP  # 43844 — one BP CNN window


class _Conv2dSame(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        dh, dw = self.dilation
        pad_h = max((math.ceil(ih / sh) - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((math.ceil(iw / sw) - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if pad_h or pad_w:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class _HarmonicStacking(nn.Module):
    def __init__(self, bins_per_semitone: int, harmonics: list[float], n_output_freqs: int) -> None:
        super().__init__()
        self.shifts = [int(round(12 * bins_per_semitone * math.log2(h))) for h in harmonics]
        self.n_output_freqs = n_output_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = []
        for shift in self.shifts:
            if shift == 0:
                channels.append(x)
            elif shift > 0:
                channels.append(F.pad(x[:, shift:, :], (0, 0, 0, shift)))
            else:
                channels.append(F.pad(x[:, :shift, :], (0, 0, -shift, 0)))
        x = torch.stack(channels, dim=1)
        return x[:, :, : self.n_output_freqs, :]


class BasicPitchModel(nn.Module):
    """Re-implementation of the Spotify Basic Pitch CNN (ICASSP 2022).

    Forward consumes mono audio at 22 050 Hz (BP's native rate), produces three
    posteriorgrams: onsets `Y_o`, contour `Y_p`, notes `Y_n` — each shape
    `(B, F, T)`.
    """

    def __init__(
        self,
        sr: int = BP_NATIVE_SR,
        hop_length: int = BP_FFT_HOP,
        annotation_semitones: int = 88,
        annotation_base: float = 27.5,
        n_harmonics: int = 8,
        contour_bins_per_semitone: int = 3,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.hop_length = hop_length
        self.contour_bins_per_semitone = contour_bins_per_semitone
        self.n_freq_bins_contour = annotation_semitones * contour_bins_per_semitone
        self.n_harmonics = n_harmonics

        # n_semitones from upstream Spotify formula at sr=22050 →
        # min(ceil(12·log2(n_harmonics))+88, floor(12·log2(sr/2/27.5))) = 103.
        # CQT2010v2 uses default filter_scale=1.0 (matches TF basic-pitch's
        # nnaudio.CQT). All BatchNorm uses eps=1e-3 to match TF/Keras default
        # (PyTorch default 1e-5 introduces ~1e-2 drift per BN layer).
        max_semitones = int(math.floor(12.0 * math.log2(0.5 * sr / annotation_base)))
        n_semitones = min(int(math.ceil(12.0 * math.log2(n_harmonics)) + annotation_semitones), max_semitones)
        self.cqt = CQT2010v2(
            sr=sr,
            fmin=annotation_base,
            hop_length=hop_length,
            n_bins=n_semitones * contour_bins_per_semitone,
            bins_per_octave=12 * contour_bins_per_semitone,
            verbose=False,
        )
        self.cqt_bn = nn.BatchNorm2d(1, eps=1e-3)
        self.contour_1 = nn.Sequential(
            nn.Conv2d(n_harmonics, 8, (39, 3), padding="same"),
            nn.BatchNorm2d(8, eps=1e-3),
            nn.ReLU(),
        )
        self.contour_2 = nn.Sequential(nn.Conv2d(8, 1, (5, 5), padding="same"), nn.Sigmoid())
        self.note_1 = nn.Sequential(
            _Conv2dSame(1, 32, (7, 7), (3, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 1, (3, 7), padding="same"),
            nn.Sigmoid(),
        )
        self.onset_1 = nn.Sequential(
            _Conv2dSame(n_harmonics, 32, (5, 5), (3, 1)),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
        )
        self.onset_2 = nn.Sequential(nn.Conv2d(32 + 1, 1, (3, 3), padding="same"), nn.Sigmoid())
        self._stacker = _HarmonicStacking(
            contour_bins_per_semitone, [0.5] + list(range(1, n_harmonics)), self.n_freq_bins_contour
        )

    @staticmethod
    def _normalised_to_db(x: torch.Tensor) -> torch.Tensor:
        log_p = 10.0 * torch.log10(x.pow(2) + 1e-10)
        log_p_min = log_p.amin(dim=(-2, -1), keepdim=True)
        offset = log_p - log_p_min
        offset_max = offset.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return offset / offset_max

    def forward_raw(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return raw posteriorgrams from BP at its native 22 050 Hz rate."""
        x = self.cqt(audio)
        x = self._normalised_to_db(x)
        x = self.cqt_bn(x.unsqueeze(1)).squeeze(1)
        x = self._stacker(x)
        x_contours = self.contour_1(x)
        x_contours = self.contour_2(x_contours).squeeze(1)
        x_notes_pre = self.note_1(x_contours.unsqueeze(1))
        x_notes = x_notes_pre.squeeze(1)
        x_onset = self.onset_1(x)
        x_onset = torch.cat([x_notes_pre, x_onset], dim=1)
        x_onset = self.onset_2(x_onset).squeeze(1)
        return {"onset": x_onset, "contour": x_contours, "note": x_notes}

    @torch.no_grad()
    def forward_windowed(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice audio into 2-s BP windows with 30-frame overlap, run forward_raw on
        each, unwrap. Matches upstream `basic_pitch.inference.window_audio_file` +
        `unwrap_output`. Input: (T,) mono at BP_NATIVE_SR. Output: dict of (F, T_bp)."""
        if audio.ndim != 1:
            raise ValueError(f"forward_windowed expects mono (T,); got {audio.shape}")

        overlap_samples = BP_OVERLAP_FRAMES * BP_FFT_HOP  # 30 * 256 = 7680
        half_overlap = overlap_samples // 2               # 3840
        hop = AUDIO_N_SAMPLES - overlap_samples           # 36164

        # Left-pad with half overlap of zeros so the first window's centre aligns at t=0.
        audio_padded = torch.cat([audio.new_zeros(half_overlap), audio], dim=0)

        # Pad on the right so an integer number of windows fits.
        n_total = audio_padded.shape[0]
        n_windows = max(1, math.ceil(n_total / hop))
        end = (n_windows - 1) * hop + AUDIO_N_SAMPLES
        if end > n_total:
            audio_padded = torch.cat([audio_padded, audio.new_zeros(end - n_total)], dim=0)

        windows = audio_padded.unfold(0, AUDIO_N_SAMPLES, hop)  # (n_windows, AUDIO_N_SAMPLES)
        raw = self.forward_raw(windows)  # {k: (n_windows, F, T_window)}

        n_drop = BP_OVERLAP_FRAMES // 2  # 15 frames each side
        # Match upstream `basic_pitch.inference.unwrap_output` — truncate to
        # int(audio_len / hop * n_frames_per_window). The naive
        # `audio.shape[0] // BP_FFT_HOP` formula uses a different (inconsistent)
        # frame contract: the BP CNN emits 172 frames per 43844-sample window
        # (not 171), so each window over-advertises stride by 188 samples
        # (~0.73 BP frames). Linear truncation keeps the time axis anchored
        # to upstream's 86 fps grid that the CSV writer assumes.
        n_frames_per_window = AUDIO_N_SAMPLES // BP_FFT_HOP + 1 - BP_OVERLAP_FRAMES
        target_frames = int(audio.shape[0] / hop * n_frames_per_window)

        out: dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            # v: (n_windows, F, T_window). Trim overlap, concat over time, truncate.
            v_trim = v[:, :, n_drop:-n_drop] if n_drop > 0 else v
            # Permute so we can concat along time: (F, n_windows * T_inner)
            F_dim = v_trim.shape[1]
            t_inner = v_trim.shape[-1]
            full = v_trim.permute(1, 0, 2).reshape(F_dim, n_windows * t_inner)
            out[k] = full[:, :target_frames]
        return out


def load_basic_pitch(weights: Optional[Path] = None) -> BasicPitchModel:
    model = BasicPitchModel()
    weights = weights or WEIGHTS_PATH
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----- canonical audio → per-voice (pitch, velocity) pipeline -----------------


@torch.no_grad()
def basic_pitch_to_voices(
    audio_bp: torch.Tensor,
    n_voices: int,
    target_frames: int,
    bp_model: BasicPitchModel | None = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The one Basic Pitch → per-voice transcription pipeline.

    `audio_bp` is mono `(T,)` **already resampled to `BP_NATIVE_SR`** (22 050 Hz).
    Resampling deliberately stays with the caller: `preprocess` resamples the
    source file directly to 22.05 kHz (keeping content up to ~10.6 kHz, which
    BP's CQT reaches), whereas the in-loop encoder only ever has 16 kHz model
    audio. Routing the cache path through 16 kHz would silently change every
    precomputed cache. `target_frames` is likewise a caller argument because
    each caller derives its own frame budget.

    Returns `(pitch, velocity)`, both `(n_voices, target_frames)` float32 on
    CPU: `pitch` in Hz (0 = silence), `velocity` in [0, 1].
    """
    if audio_bp.ndim != 1:
        raise ValueError(f"basic_pitch_to_voices expects mono (T,); got {tuple(audio_bp.shape)}")

    bp_model = bp_model or load_basic_pitch().to(audio_bp.device)

    # BP forward (windowed — required for CSV parity and memory).
    post = bp_model.forward_windowed(audio_bp)  # {onset, contour, note} (F, T_bp)

    # Note extraction at BP-native rate, BP CLI defaults.
    frames_np = post["note"].T.cpu().numpy()       # (T_bp, 88)
    onsets_np = post["onset"].T.cpu().numpy()      # (T_bp, 88)
    contours_np = post["contour"].T.cpu().numpy()  # (T_bp, 264)

    note_events = output_to_notes_polyphonic(
        frames=frames_np, onsets=onsets_np,
        onset_thresh=0.5, frame_thresh=0.3, min_note_len=11,
        infer_onsets=True, melodia_trick=True, energy_tol=11,
        max_freq=max_freq, min_freq=min_freq,
    )
    note_events = get_pitch_bends(contours_np, note_events, n_bins_tolerance=25)
    # Drop overlapping pitch bends BEFORE allocation (matches BP `multiple_pitch_bends=False`).
    note_events = drop_overlapping_pitch_bends(note_events)

    # Resample note grid (88, T_bp) → (88, T_target) for per-frame velocity.
    note_grid = post["note"].unsqueeze(0).unsqueeze(0)  # (1, 1, 88, T_bp)
    note_grid_target = F.interpolate(
        note_grid, size=(post["note"].shape[0], target_frames),
        mode="bilinear", align_corners=True,
    )[0, 0]  # (88, T_target)

    bp_to_target_ratio = target_frames / max(1, post["note"].shape[-1])
    return allocate_to_voices(
        note_events, note_grid_target.cpu(), n_voices,
        target_frames, bp_to_target_ratio,
    )


# ----- top-level wrapper -----------------------------------------------------


class PitchEncoder(nn.Module):
    """Pitch encoder with two Basic Pitch backends.

    `source="basic_pitch"`: runs the frozen Basic Pitch CNN in-loop on the input
        audio via `basic_pitch_to_voices` — the same pipeline `preprocess`
        caches, one clip at a time. Slow (note extraction is sequential numpy),
        so it exists for inference on unseen audio, not for training.

    `source="cached_basic_pitch"` (used for training): no in-loop pitch model.
        The caller passes precomputed per-voice tensors — `pitch_hint` in Hz and
        `velocity_hint`, both shape (B, n_voices, target_frames) — which are
        truncated to the audio's frame count and returned as-is. The cache is
        written by `polyddsp/preprocess.py`, which runs the identical pipeline
        once per whole file, so it sees note context across crop boundaries.

    Explicit pitch and velocity hints take precedence for either source. This
    lets inference transcribe the input live with a caller-selected model and
    pass the resulting tensors directly, without reading a cache file.

    Both sources return an empty `bp_post` dict; the key is retained because
    `polyddsp.py` forwards it.
    """

    _CACHED_SOURCES = ("cached_basic_pitch",)
    _VALID_SOURCES = ("basic_pitch", "cached_basic_pitch")

    def __init__(
        self,
        sr: int = 16_000,
        n_voices: int = 1,
        target_frame_hop: int = 64,
        source: str = "basic_pitch",
    ) -> None:
        super().__init__()
        if source not in self._VALID_SOURCES:
            raise ValueError(f"unknown pitch source: {source!r}")
        self.sr = sr
        self.n_voices = n_voices
        self.target_frame_hop = target_frame_hop
        self.source = source
        if source == "basic_pitch":
            self.bp = load_basic_pitch()
            from torchaudio.transforms import Resample
            self._to_bp = Resample(orig_freq=sr, new_freq=BP_NATIVE_SR)
        else:
            self.bp = None
            self._to_bp = None

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        pitch_hint: torch.Tensor | None = None,
        velocity_hint: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (pitch_hint is not None or velocity_hint is not None
                or self.source in self._CACHED_SOURCES):
            return self._forward_cached(audio, pitch_hint, velocity_hint)
        return self._forward_basic_pitch(audio)

    def _forward_cached(
        self,
        audio: torch.Tensor,
        pitch_hint: torch.Tensor | None,
        velocity_hint: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        target_frames = audio.shape[-1] // self.target_frame_hop
        if pitch_hint is None or velocity_hint is None:
            raise RuntimeError(
                f"PitchEncoder(source={self.source!r}) requires pitch_hint and velocity_hint; "
                "the dataset must return a dict with 'pitch' and 'velocity' (precompute via "
                "`polyddsp-preprocess experiment=<name>`)"
            )
        pitch = pitch_hint[..., :target_frames].to(device=audio.device, dtype=torch.float32)
        velocity = velocity_hint[..., :target_frames].to(device=audio.device, dtype=torch.float32)
        return {"pitch": pitch, "velocity": velocity, "bp_post": {}}

    def _forward_basic_pitch(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the same pipeline `preprocess` caches, one clip at a time.

        Loops over the batch because note extraction is sequential per-item
        numpy, resampling each item to BP's native rate first. Because this only
        ever sees the audio handed to it, a note that crosses the crop boundary
        is truncated at that boundary — the precomputed cache, which transcribes
        whole files, does not have that limitation. `n_voices == 1` is not
        special-cased: it is just V=1 of the same allocator.
        """
        if audio.ndim != 2:
            raise ValueError(f"PitchEncoder expects batched audio (B, T); got {tuple(audio.shape)}")
        target_frames = audio.shape[-1] // self.target_frame_hop
        audio_bp = self._to_bp(audio)
        pitches, velocities = [], []
        for b in range(audio_bp.shape[0]):
            p, v = basic_pitch_to_voices(
                audio_bp[b],
                n_voices=self.n_voices,
                target_frames=target_frames,
                bp_model=self.bp,
            )
            pitches.append(p)
            velocities.append(v)
        pitch = torch.stack(pitches).to(device=audio.device, dtype=torch.float32)
        velocity = torch.stack(velocities).to(device=audio.device, dtype=torch.float32)
        return {"pitch": pitch, "velocity": velocity, "bp_post": {}}
