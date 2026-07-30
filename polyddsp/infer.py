"""Render audio from a trained checkpoint: file in, resynthesised file out.

    python -m polyddsp.infer ckpt=<run>/best.pt input=x.wav out=y.wav [device=cuda]

Basic Pitch runs in-process, so there is no precompute step — point this at any
audio file. Both `key=value` (shown above, and in the README) and the usual
`--key value` spellings work; see `_normalise_argv`.

**Why argparse and not Hydra**, unlike `train.py` / `eval.py`:

1. `configs/config.yaml` sets `hydra.run.dir: ${run.out_dir}`, so `@hydra.main`
   would create a fresh timestamped run directory and chdir into it merely to
   render a wav — and `out=y.wav` would then resolve against that new cwd
   instead of the user's.
2. For inference the *run directory's* saved `config.yaml` must be
   authoritative. Composing defaults from `configs/` would silently rebuild a
   different architecture than the checkpoint was trained with unless every
   override were retyped on the CLI.
3. A saved run config can contain `${now:%Y%m%d_%H%M%S}` in `run.name`, a
   resolver that only exists inside a Hydra app. So never call
   `OmegaConf.to_container(cfg, resolve=True)` on a loaded run config.
   `PolyDDSP.from_cfg` only reads non-interpolating keys, which is why passing
   the raw loaded config to it is safe.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from polyddsp.model.polyddsp import PolyDDSP
from polyddsp.preprocess import load_resample

_LIVE_BP_SOURCES = ("cached_basic_pitch", "basic_pitch")


def _normalise_argv(argv: list[str]) -> list[str]:
    """Rewrite Hydra-style `key=value` tokens as `--key value`.

    The README documents the `key=value` form (it reads the same as the
    `train.py` / `eval.py` invocations next to it), but this module is plain
    argparse. Tokens that already start with `-`, and bare positional tokens,
    pass through untouched, so `--out y.wav` and `out=y.wav` are interchangeable.
    Splits on the *first* `=` only, so values may contain `=`.
    """
    out: list[str] = []
    for tok in argv:
        if not tok.startswith("-") and "=" in tok:
            key, value = tok.split("=", 1)
            out += [f"--{key}", value]
        else:
            out.append(tok)
    return out


def load_run(ckpt_path: str | Path, device: str = "cpu") -> tuple[PolyDDSP, DictConfig]:
    """Rebuild the model a checkpoint was trained as, from the run's own config.

    `polyddsp.train` writes the resolved config to `config.yaml` in the run
    directory, beside `best.pt` / `last.pt` / `top_*.pt` — so the config is read
    from the checkpoint's own parent directory, never composed from `configs/`.
    Handles both `{"model": state_dict}` (what `train.py` saves) and a bare
    state dict. Returns an `eval()`-mode model already on `device`.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"no config.yaml next to the checkpoint ({cfg_path}).\n"
            "polyddsp.train writes the resolved config into the run directory; the model "
            "architecture cannot be reconstructed without it. Copy the run's config.yaml "
            f"into {ckpt_path.parent} and retry."
        )

    # Deliberately unresolved: `run.name` may hold `${now:...}`, a Hydra-only resolver.
    cfg = OmegaConf.load(cfg_path)
    model = PolyDDSP.from_cfg(cfg)

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def pitch_features(
    audio_path: str | Path,
    cfg: DictConfig,
    target_frames: int,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Transcribe `audio_path` to per-voice `{pitch, velocity}` conditioning.

    Runs Basic Pitch in-process via `polyddsp.model.pitch.basic_pitch_to_voices`
    — the same function that writes the training cache — so inference needs no
    `.f0.pt` sidecar and never reads one.

    The file is loaded *directly* at `BP_NATIVE_SR` (22 050 Hz) rather than
    resampled up from the model's rate: 16 kHz → 22.05 kHz would throw away the
    band above 8 kHz that BP's CQT reaches, and the training cache was built from
    the 22.05 kHz load. Matching it here is what makes infer-time pitch agree
    with training.

    Returns `{"pitch": (1, V, target_frames), "velocity": (1, V, target_frames)}`.
    """
    from polyddsp.model.pitch import BP_NATIVE_SR, basic_pitch_to_voices

    source = cfg.experiment.model.get("pitch_source", "basic_pitch")
    if source not in _LIVE_BP_SOURCES:
        raise SystemExit(
            f"pitch_source={source!r} is not supported by polyddsp.infer; "
            f"expected one of {_LIVE_BP_SOURCES}"
        )

    audio_bp = load_resample(Path(audio_path), BP_NATIVE_SR).to(device)
    pitch, velocity = basic_pitch_to_voices(
        audio_bp,
        n_voices=cfg.experiment.model.n_voices,
        target_frames=target_frames,
    )
    return {
        "pitch": pitch.unsqueeze(0).to(device),
        "velocity": velocity.unsqueeze(0).to(device),
    }


def render_file(
    ckpt: str | Path,
    audio_path: str | Path,
    out_path: str | Path,
    device: str | None = None,
    chunk_frames: int | None = None,
) -> None:
    """Resynthesise one audio file through a checkpoint and write the result.

    The input is loaded twice — once at `cfg.model.sr` for the model, once at
    Basic Pitch's native rate for transcription (see `pitch_features`), mirroring
    what `preprocess.py` does. Output is peak-normalised, since DDSP output level
    depends on the learned reverb/dry balance and is not calibrated to the input.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_run(ckpt, device)

    sr = int(cfg.model.sr)
    hop = int(cfg.model.frame_hop)
    audio = load_resample(Path(audio_path), sr)
    n_frames = audio.shape[-1] // hop
    if n_frames == 0:
        raise SystemExit(
            f"{audio_path} is shorter than one frame ({hop} samples at {sr} Hz)"
        )
    audio = audio[: n_frames * hop]  # synthesis works on whole frames

    if chunk_frames is None:
        # One training clip's worth of frames: the block the decoder GRU was fit on.
        chunk_frames = int(float(cfg.model.clip_seconds) * sr) // hop

    source = cfg.experiment.model.get("pitch_source", "basic_pitch")
    if source == "cached_basic_pitch":
        feats = pitch_features(audio_path, cfg, n_frames, device)
    elif source == "basic_pitch":
        # `PitchEncoder(source="basic_pitch")` transcribes in-loop from the
        # model-rate audio and ignores any hint, so transcribing here too would
        # just be wasted work.
        print(
            "note: this run's pitch_source is the in-loop 'basic_pitch' encoder — Basic Pitch "
            f"runs inside the model on {sr} Hz audio, not on a 22.05 kHz load"
        )
        feats = {}
    else:
        raise SystemExit(
            f"pitch_source={source!r} is not supported by polyddsp.infer; "
            f"expected one of {_LIVE_BP_SOURCES}"
        )

    out = model.render(
        audio.unsqueeze(0).to(device),
        chunk_frames=chunk_frames,
        **feats,
    )

    y = out[0].detach().cpu()
    peak = y.abs().max()
    if peak > 0:
        y = y / peak
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y.numpy(), sr)

    n_blocks = max(1, math.ceil(n_frames / chunk_frames))
    print(
        f"rendered {n_frames * hop / sr:.2f}s @ {sr} Hz "
        f"({n_frames} frames, {n_blocks} block(s), V={cfg.experiment.model.n_voices}, "
        f"device={device}) -> {out_path}"
    )


def main(argv: list[str] | None = None) -> None:
    import sys

    p = argparse.ArgumentParser(
        prog="python -m polyddsp.infer",
        description="Resynthesise an audio file through a trained PolyDDSP checkpoint.",
    )
    p.add_argument("--ckpt", required=True, help="path to best.pt / last.pt / top_*.pt")
    p.add_argument("--input", required=True, help="input audio file (any soundfile format)")
    p.add_argument("--out", required=True, help="output .wav path")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument(
        "--chunk-frames", "--chunk_frames", dest="chunk_frames", type=int, default=None,
        help="frames of audio synthesised per block (default: cfg.model.clip_seconds worth)",
    )
    args = p.parse_args(_normalise_argv(list(sys.argv[1:] if argv is None else argv)))

    render_file(
        ckpt=args.ckpt,
        audio_path=args.input,
        out_path=args.out,
        device=args.device,
        chunk_frames=args.chunk_frames,
    )


if __name__ == "__main__":
    main()
