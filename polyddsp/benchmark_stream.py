"""Benchmark checkpoint synthesis from controls, independently of pitch extraction."""
from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import torch

from polyddsp.infer import load_run
from polyddsp.model.parity_ops import midi_to_hz
from polyddsp.model.streaming import StreamingSynthesizer


def benchmark(
    synth: StreamingSynthesizer,
    controls: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    *,
    iterations: int = 200,
    warmup: int = 30,
) -> dict:
    """Time actual stateful calls, synchronizing CUDA to include device execution.

    This measures synthetic-control inference, not audio I/O, feature analysis,
    transcription, or sound quality. No model weights or files are changed.
    """
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    pitch = controls[0]
    if pitch.shape[-1] < 1:
        raise ValueError("benchmark controls must contain at least one frame")
    device = pitch.device
    state = None
    for _ in range(warmup):
        _, state = synth.process(*controls, state)
    durations = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        _, state = synth.process(*controls, state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((perf_counter() - start) * 1000.0)
    block_ms = 1000.0 * pitch.shape[-1] * synth.frame_hop / synth.sample_rate
    return {
        "device": str(device), "cpu_threads": torch.get_num_threads(),
        "voices": pitch.shape[1], "block_frames": pitch.shape[-1], "block_ms": block_ms,
        "iterations": iterations, "median_ms": float(np.median(durations)),
        "p99_ms": float(np.percentile(durations, 99)), "max_ms": max(durations),
        "deadline_misses": sum(t > block_ms for t in durations),
        "scope": "synthesis only; synthetic controls; excludes audio I/O and feature extraction",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="checkpoint with an adjacent config.yaml")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--block-frames", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU intra-op threads")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.block_frames, args.threads, args.iterations) < 1 or args.warmup < 0:
        parser.error("block-frames, threads, and iterations must be positive; warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    model, _ = load_run(args.ckpt, args.device)
    synth = StreamingSynthesizer.from_model(model, noise_seed=0).eval()
    frames = args.block_frames
    notes = torch.linspace(48.0, 72.0, model.n_voices, device=args.device)
    pitch = midi_to_hz(notes).reshape(1, model.n_voices, 1).expand(-1, -1, frames).contiguous()
    velocity = torch.full_like(pitch, 0.7)
    loudness = pitch.new_full((1, frames), 0.6)
    z = pitch.new_zeros(1, frames, model.decoder.z_dim) if model.use_z else None
    result = benchmark(synth, (pitch, velocity, loudness, z), iterations=args.iterations, warmup=args.warmup)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
