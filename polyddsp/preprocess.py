"""Precompute Basic Pitch note annotations for each audio file in a dataset.

For each `<file>` matching `--glob`, runs Basic Pitch, extracts note events, and
allocates them to `--n-voices` slots. The resulting `{pitch, velocity}` tensors
(each (V, T) at `audio_len // hop` frames of the target sample rate) are cached
beside the audio by default. With `--output-dir`, paths relative to `--root` are
preserved beneath a separate cache root. Idempotent: skips files whose cache
mtime is newer than the source audio mtime.

Usage:
    uv run python -m polyddsp.preprocess --root <dir> [--output-dir <dir>] \\
        [--glob '**/*.mp3'] \\
        [--sample-rate 16000] [--hop 64] [--n-voices 6] [--device cuda] \\
        [--min-freq HZ] [--max-freq HZ]

The transcription itself lives in `polyddsp.model.pitch.basic_pitch_to_voices`,
which the in-loop `PitchEncoder(source="basic_pitch")` also calls — there is
exactly one Basic Pitch pipeline in this repo.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from polyddsp.model.pitch import BasicPitchModel


def cache_path_for(
    audio_path: Path,
    suffix: str,
    *,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Return a cache path, optionally mirrored beneath a separate output root."""
    target = audio_path
    if output_dir is not None:
        if root is None:
            raise ValueError("root is required when output_dir is provided")
        try:
            relative = audio_path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"audio path {audio_path} is outside dataset root {root}") from exc
        target = output_dir / relative
    return target.with_suffix(target.suffix + f".{suffix}.f0.pt")


def load_resample(path: Path, target_sr: int) -> torch.Tensor:
    """Read an audio file as mono float32 and resample to `target_sr`."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    audio = torch.from_numpy(wav)
    if sr != target_sr:
        audio = AF.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio


def bp_cache_suffix(n_voices: int, sample_rate: int, target_hop: int) -> str:
    return f"bp_v{n_voices}_sr{sample_rate}_hop{target_hop}"


def resolve_pitch_cache(cfg) -> tuple[str | None, str | None, int | None]:
    """Map a Hydra config to `RawAudioDataset`'s (kind, suffix, n_voices) cache args.

    Returns `(None, None, None)` when the experiment's pitch source runs in-loop
    rather than reading a precomputed cache.
    """
    if cfg.experiment.model.get("pitch_source") != "cached_basic_pitch":
        return None, None, None
    n_voices = cfg.experiment.model.n_voices
    suffix = bp_cache_suffix(n_voices, cfg.model.sr, cfg.model.frame_hop)
    return "basic_pitch", suffix, n_voices


def precompute_one_bp(
    audio_path: Path,
    sample_rate: int,
    target_hop: int,
    n_voices: int,
    device: str = "cpu",
    bp_model: Optional["BasicPitchModel"] = None,
    min_freq: float | None = None,
    max_freq: float | None = None,
    root: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, bool]:
    """Cache (pitch, velocity) tensors at target rate. Returns (cache_path, recomputed).

    Thin I/O wrapper around `polyddsp.model.pitch.basic_pitch_to_voices` — the
    same function the in-loop `PitchEncoder(source="basic_pitch")` calls, so the
    cache and the live encoder cannot diverge. Pass `bp_model` to reuse one
    loaded CNN (and its CQT kernels) across files.
    """
    from polyddsp.model.pitch import BP_NATIVE_SR, basic_pitch_to_voices

    suffix = bp_cache_suffix(n_voices, sample_rate, target_hop)
    cache = cache_path_for(audio_path, suffix, root=root, output_dir=output_dir)
    if cache.exists() and cache.stat().st_mtime > audio_path.stat().st_mtime:
        return cache, False

    # Resample the source file *directly* to BP's native rate: going via the
    # model rate would discard everything above 8 kHz that BP's CQT reaches.
    audio_22k = load_resample(audio_path, BP_NATIVE_SR).to(device)

    # Frame-count budget at target rate.
    n_target_samples = int(round(audio_22k.shape[-1] * sample_rate / BP_NATIVE_SR))
    target_frames = n_target_samples // target_hop

    pitch, velocity = basic_pitch_to_voices(
        audio_22k,
        n_voices=n_voices,
        target_frames=target_frames,
        bp_model=bp_model,
        min_freq=min_freq,
        max_freq=max_freq,
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pitch": pitch.contiguous(), "velocity": velocity.contiguous()}, cache)
    return cache, True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--output-dir", "--output_dir",
        dest="output_dir",
        type=Path,
        default=None,
        help=("Optional cache root; the source directory structure is mirrored beneath it. "
              "By default caches are written beside the audio files."),
    )
    p.add_argument("--glob", default="**/*.mp3")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hop", type=int, default=64)
    p.add_argument("--n-voices", type=int, default=6,
                   help="V slots for cached_basic_pitch")
    p.add_argument("--min-freq", type=float, default=None,
                   help="Drop transcribed notes below this Hz (e.g. rumble / handling noise).")
    p.add_argument("--max-freq", type=float, default=None,
                   help="Drop transcribed notes at or above this Hz.")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args(argv)

    files = sorted(args.root.glob(args.glob))
    if not files:
        print(f"no files in {args.root} matching {args.glob}", file=sys.stderr)
        sys.exit(1)
    suffix = bp_cache_suffix(args.n_voices, args.sample_rate, args.hop)
    print(f"found {len(files)} files; caching BP allocation with suffix={suffix}")

    # Load the CNN (and build its CQT kernels) exactly once — rebuilding
    # CQT2010v2 per file costs ~1 s each, which dominates on large corpora.
    from polyddsp.model.pitch import load_basic_pitch

    t0 = time.perf_counter()
    bp_model = load_basic_pitch().to(args.device)
    print(f"loaded Basic Pitch in {time.perf_counter() - t0:.2f}s (reused for all files)")

    t0 = time.perf_counter()
    n_written = 0
    progress = tqdm(files, desc="preprocess", unit="file", dynamic_ncols=True, position=0)
    # A second, text-only line keeps long filenames from changing the main
    # bar's width. Unlike tqdm.write(), it is replaced in place and erased when
    # preprocessing finishes instead of leaving one permanent line per file.
    status = tqdm(total=0, bar_format="{desc}", leave=False, dynamic_ncols=True, position=1)
    try:
        for f in progress:
            cache, recomputed = precompute_one_bp(
                f, args.sample_rate, args.hop, args.n_voices, args.device,
                bp_model=bp_model, min_freq=args.min_freq, max_freq=args.max_freq,
                root=args.root, output_dir=args.output_dir,
            )
            n_written += int(recomputed)
            marker = "wrote" if recomputed else "skip "
            status.set_description_str(f"[{marker}] {f.relative_to(args.root)}", refresh=True)
    finally:
        status.close()
    print(f"done: {n_written}/{len(files)} recomputed in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
