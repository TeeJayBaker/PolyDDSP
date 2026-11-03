"""Shared configuration for optional instrument entry points."""
from __future__ import annotations

import argparse
from importlib import import_module


def optional_module(name: str):
    try:
        return import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Optional MIDI dependency {name!r} is unavailable. Run `uv sync --extra midi`; "
            "live audio also requires the native PortAudio library."
        ) from exc


def add_model_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument("--ckpt", required=required)
    parser.add_argument("--soundfont", required=required, help="path to a user-supplied guitar .sf2")
    parser.add_argument("--bank", type=int, default=0)
    parser.add_argument("--preset", type=int, default=24, help="zero-based SF2 preset; GM 24 is nylon guitar")
    parser.add_argument("--reference-gain", type=float, default=0.2)
    parser.add_argument("--output-gain", type=float, default=1.0)
    parser.add_argument("--pitch-bend-range", type=float, default=2.0, help="semitones, shared by patch and controls")
    parser.add_argument("--device", default="cpu", help="model device, e.g. cpu or cuda:0")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--block-frames", type=int, default=4, help="control frames per call; 4 = 16 ms at defaults")
    parser.add_argument("--noise-seed", type=int, default=0)


def make_instrument(args):
    import torch

    from polyddsp.infer import load_run
    from polyddsp.midi_instrument.engine import MidiInstrument
    from polyddsp.midi_instrument.renderer import FluidSynthRenderer

    if args.threads < 1 or args.block_frames < 1:
        raise ValueError("threads and block-frames must be positive")
    torch.set_num_threads(args.threads)
    model, _ = load_run(args.ckpt, args.device, require_streaming=True)
    renderer = FluidSynthRenderer(
        args.soundfont, model.sr, model.n_voices, bank=args.bank, preset=args.preset,
        gain=args.reference_gain, pitch_bend_range=args.pitch_bend_range,
    )
    try:
        return MidiInstrument(
            model, renderer, pitch_bend_range=args.pitch_bend_range,
            output_gain=args.output_gain, noise_seed=args.noise_seed,
        )
    except Exception:
        renderer.close()
        raise
