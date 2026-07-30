"""Audible audit of the BP cache: side-by-side original + sinewave resynth.

Runs `preprocess.precompute_one_bp` on a given audio file (cached if present),
then synthesises a per-voice sin(2π·∫f0/sr) bank weighted by `velocity`. Sums
across voices. Writes:

    <out_dir>/<stem>__orig.wav   — original audio (16 kHz mono)
    <out_dir>/<stem>__sines.wav  — resynth (16 kHz mono)
    <out_dir>/<stem>__ab.wav     — stereo: L=orig, R=sines (drop into any DAW for A/B)
    <out_dir>/<stem>__roll.png   — quick piano-roll plot (one row per voice)

Also prints diagnostic stats: per-voice frame-occupancy, midi range, distinct-note
count. The aim is to verify the BP transcription is sane *before* the synth/decoder
ever sees it — if these sines don't sound like the input's notes, the cache is
the problem, not the model.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from polyddsp.model.parity_ops import hz_to_midi
from polyddsp.preprocess import load_resample, precompute_one_bp


def _upsample_linear(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear upsample along the last dim. `x` shape: (V, T_frames)."""
    return torch.nn.functional.interpolate(
        x.unsqueeze(0), size=target_len, mode="linear", align_corners=False
    ).squeeze(0)


def synth_sines(
    pitch: torch.Tensor,       # (V, T_frames) Hz
    velocity: torch.Tensor,    # (V, T_frames) [0,1]
    sr: int,
    frame_hop: int,
    chunk_size: int = 4000,
) -> torch.Tensor:
    """Sum-of-voices sinewave at sample rate. Each voice is sin(2π·∫f0)·vel."""
    V, T_frames = pitch.shape
    T_samples = T_frames * frame_hop

    pitch_s = _upsample_linear(pitch, T_samples)        # (V, T_samples)
    vel_s = _upsample_linear(velocity, T_samples)       # (V, T_samples)

    # Crossfade in/out at note boundaries to avoid clicks: a 4 ms ramp on velocity.
    # Velocity already does this implicitly because BP frames are 4 ms — leave it.

    # Per-sample angular increment then chunked cumsum (mod 2π) — same recipe
    # as polyddsp.model.additive.angular_cumsum but in numpy is fine since this
    # script is one-shot CPU.
    omegas = pitch_s * (2.0 * math.pi / sr)             # (V, T_samples)
    # Chunked cumsum to avoid fp32 phase drift on long files.
    pad = (-T_samples) % chunk_size
    if pad:
        omegas = torch.cat([omegas, omegas.new_zeros(V, pad)], dim=-1)
    L = omegas.shape[-1]
    n_chunks = L // chunk_size
    chunks = omegas.reshape(V, n_chunks, chunk_size)
    phase = torch.cumsum(chunks, dim=-1)
    chunk_ends = phase[..., -1] % (2.0 * math.pi)
    cum_offsets = torch.cumsum(chunk_ends, dim=-1) % (2.0 * math.pi)
    cum_offsets = torch.roll(cum_offsets, shifts=1, dims=-1)
    cum_offsets[..., 0] = 0
    phase = (phase + cum_offsets.unsqueeze(-1)) % (2.0 * math.pi)
    phase = phase.reshape(V, L)
    if pad:
        phase = phase[..., :T_samples]

    sin_phase = torch.sin(phase)
    # Mute samples where pitch == 0 (silence between notes); keeps noise floor clean.
    sin_phase = torch.where(pitch_s > 0, sin_phase, torch.zeros_like(sin_phase))
    voice_audio = sin_phase * vel_s                      # (V, T_samples)
    return voice_audio.sum(dim=0)                        # (T_samples,)


def piano_roll_png(
    pitch: torch.Tensor, velocity: torch.Tensor, png_path: Path
) -> None:
    """One subplot per voice: midi-vs-frame heatmap of velocity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V, T = pitch.shape
    midi = hz_to_midi(pitch)  # (V, T)

    fig, axes = plt.subplots(V, 1, figsize=(14, 1.8 * V), sharex=True)
    if V == 1:
        axes = [axes]
    for v in range(V):
        ax = axes[v]
        m = midi[v].numpy()
        a = velocity[v].numpy()
        ax.scatter(np.arange(T), m, c=a, cmap="viridis", s=2, vmin=0, vmax=max(1.0, float(a.max() or 1.0)))
        ax.set_ylim(20, 100)
        ax.set_ylabel(f"v{v}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame (target rate)")
    fig.suptitle(f"BP cache — pitch (midi) coloured by velocity (V={V}, T={T})")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def diagnostics(pitch: torch.Tensor, velocity: torch.Tensor) -> str:
    V, T = pitch.shape
    lines = [f"Cache shape: V={V}, T={T} (target frames)"]
    midi = hz_to_midi(pitch)
    for v in range(V):
        active = (pitch[v] > 0)
        n_active = int(active.sum().item())
        if n_active == 0:
            lines.append(f"  v{v}: empty")
            continue
        m = midi[v][active]
        a = velocity[v][active]
        # Count distinct contiguous note runs.
        on = active.numpy().astype(np.int8)
        edges = np.diff(np.concatenate([[0], on, [0]]))
        n_notes = int((edges == 1).sum())
        lines.append(
            f"  v{v}: occupancy={n_active}/{T} ({100*n_active/T:.1f}%), "
            f"distinct_notes={n_notes}, midi={float(m.min()):.1f}–{float(m.max()):.1f}, "
            f"vel mean={float(a.mean()):.3f} (max={float(a.max()):.3f})"
        )
    return "\n".join(lines)


def _run_via_dataset(args, cache_path: Path) -> None:
    """Use RawAudioDataset to crop audio + cache at training-time offsets, resynth.

    This tests the EXACT slicing the trainer does. If the sines track the
    audio in each crop, train-time alignment is correct.
    """
    from polyddsp.data import RawAudioDataset

    root = args.audio.parent
    glob = args.audio.name  # only this one file
    ds = RawAudioDataset(
        root=str(root),
        split="train",  # train split = all files when there's only one
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        seed=0,
        file_glob=glob,
        pitch_cache_kind="basic_pitch",
        pitch_cache_suffix=cache_path.name.split(".f0.pt")[0].split(args.audio.suffix + ".")[-1],
        f0_hop=args.target_hop,
        n_voices=args.n_voices,
    )
    if len(ds.active) == 0:
        # Single-file split may land in val. Try val split.
        ds = RawAudioDataset(
            root=str(root), split="val", sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds, seed=0,
            file_glob=glob, pitch_cache_kind="basic_pitch",
            pitch_cache_suffix=cache_path.name.split(".f0.pt")[0].split(args.audio.suffix + ".")[-1],
            f0_hop=args.target_hop, n_voices=args.n_voices,
        )

    print(f"\n--- via-dataset mode: {len(ds.active)} file(s), {args.n_crops} crops ---")
    stem = args.audio.stem
    for crop_i in range(args.n_crops):
        sample = ds[crop_i]  # deterministic in val, stochastic in train
        if not isinstance(sample, dict) or "pitch" not in sample:
            print(f"  crop {crop_i}: dataset returned {type(sample).__name__}, skipping")
            continue
        clip = sample["audio"]
        pitch = sample["pitch"]
        velocity = sample["velocity"]
        sines = synth_sines(pitch, velocity, args.sample_rate, args.target_hop)
        n = min(clip.shape[-1], sines.shape[-1])
        clip_n = clip[:n]
        sines_n = sines[:n]
        peak = float(clip_n.abs().max().clamp_min(1e-9))
        peak_s = float(sines_n.abs().max())
        if peak_s > 1e-9:
            sines_n = sines_n * (peak / peak_s)
        stereo = np.stack([clip_n.numpy(), sines_n.numpy()], axis=-1)
        out_path = args.out_dir / f"{stem}__cropped_{crop_i:02d}__ab.wav"
        sf.write(out_path, stereo, args.sample_rate)
        # Per-voice activity in this crop:
        slot_live = (pitch > 0).any(dim=-1)
        n_live = int(slot_live.sum().item())
        active_pitches = []
        for v in range(pitch.shape[0]):
            if slot_live[v]:
                m = hz_to_midi(pitch[v][pitch[v] > 0])
                active_pitches.append(f"v{v}=[{float(m.min()):.0f}–{float(m.max()):.0f}]")
        print(f"  crop {crop_i}: {n_live}/{pitch.shape[0]} voices live "
              f"{' '.join(active_pitches)} -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True, help="Input audio file (any sample rate).")
    p.add_argument("--out-dir", type=Path, required=True, help="Where to write WAVs/PNG.")
    p.add_argument("--n-voices", type=int, default=6)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--target-hop", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-roll", action="store_true", help="Skip the piano-roll PNG.")
    p.add_argument(
        "--via-dataset",
        action="store_true",
        help="Crop via RawAudioDataset (same path as trainer). Tests train-time alignment.",
    )
    p.add_argument("--clip-seconds", type=float, default=4.0,
                   help="Crop duration when --via-dataset.")
    p.add_argument("--n-crops", type=int, default=3,
                   help="How many distinct crops to render in --via-dataset mode.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build / read BP cache.
    cache_path, recomputed = precompute_one_bp(
        args.audio, args.sample_rate, args.target_hop, args.n_voices, device=args.device
    )
    print(f"[{'wrote' if recomputed else 'skip '}] {cache_path}")

    if args.via_dataset:
        return _run_via_dataset(args, cache_path)

    blob = torch.load(cache_path, map_location="cpu", weights_only=True)
    pitch = blob["pitch"].to(torch.float32)
    velocity = blob["velocity"].to(torch.float32)
    print(diagnostics(pitch, velocity))

    # 2. Reload the original audio at sample_rate (mono) for side-by-side.
    orig = load_resample(args.audio, args.sample_rate).to(torch.float32)
    # The cache time grid ends at exactly T_frames * target_hop samples, which
    # may differ from `orig.shape[0]` by < target_hop due to the resampling
    # round in precompute_one_bp. Trim both to the same length.
    T_samples_cache = pitch.shape[-1] * args.target_hop
    n = min(orig.shape[0], T_samples_cache)
    orig = orig[:n]

    # 3. Synthesise sine bank.
    sines = synth_sines(pitch, velocity, args.sample_rate, args.target_hop)
    sines = sines[:n]
    # Peak-normalise the sine output to match the original's peak (so A/B is
    # at the same loudness). If empty, leave as-is.
    peak_orig = float(orig.abs().max().clamp_min(1e-9))
    peak_sines = float(sines.abs().max())
    if peak_sines > 1e-9:
        sines = sines * (peak_orig / peak_sines)

    # 4. Write outputs.
    stem = args.audio.stem
    out_orig = args.out_dir / f"{stem}__orig.wav"
    out_sines = args.out_dir / f"{stem}__sines.wav"
    out_ab = args.out_dir / f"{stem}__ab.wav"
    sf.write(out_orig, orig.numpy(), args.sample_rate)
    sf.write(out_sines, sines.numpy(), args.sample_rate)
    stereo = np.stack([orig.numpy(), sines.numpy()], axis=-1)
    sf.write(out_ab, stereo, args.sample_rate)
    print(f"orig  -> {out_orig}")
    print(f"sines -> {out_sines}")
    print(f"A/B   -> {out_ab}  (L=orig, R=sines)")

    if not args.no_roll:
        png = args.out_dir / f"{stem}__roll.png"
        piano_roll_png(pitch, velocity, png)
        print(f"roll  -> {png}")


if __name__ == "__main__":
    main()
