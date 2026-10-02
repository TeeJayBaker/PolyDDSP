"""Render audio from a trained checkpoint: file in, resynthesised file out.

    polyddsp-infer --ckpt <run>/best.pt --input x.wav --out y.wav \
        [--pitch-encoder basic-pitch]

The selected pitch encoder runs in-process, so there is no precompute step —
point this at any audio file. The usual `--key value` spelling shown above and
`key=value` are both accepted; see `_normalise_argv`.

**Why argparse and not Hydra**, unlike `train.py` / `eval.py`:

1. Rendering is not a Hydra job: it should not compose an experiment config or
   initialise Hydra's job logging/runtime merely to turn one file into another.
   Plain argparse also keeps input/output paths relative to the caller's cwd.
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

_PITCH_ENCODERS = ("basic-pitch", "neutone-amt")


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

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model.to(device).eval(), cfg


def pitch_features(
    audio_path: str | Path,
    cfg: DictConfig,
    target_frames: int,
    device: str = "cpu",
    pitch_encoder: str = "basic-pitch",
    pitch_encoder_checkpoint: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Transcribe `audio_path` to per-voice `{pitch, velocity}` conditioning.

    Uses the same Basic Pitch or Neutone AMT transcription path as preprocessing,
    but returns the tensors directly instead of writing a cache sidecar.

    Returns `{"pitch": (1, V, target_frames), "velocity": (1, V, target_frames)}`.
    """
    if pitch_encoder == "basic-pitch":
        from polyddsp.model.pitch import BP_NATIVE_SR, basic_pitch_to_voices

        # Load directly at BP's native rate rather than resampling the model-rate
        # audio, which would have already discarded everything above 8 kHz.
        audio_bp = load_resample(Path(audio_path), BP_NATIVE_SR).to(device)
        pitch, velocity = basic_pitch_to_voices(
            audio_bp,
            n_voices=cfg.experiment.model.n_voices,
            target_frames=target_frames,
        )
    elif pitch_encoder == "neutone-amt":
        if pitch_encoder_checkpoint is None:
            raise SystemExit(
                "--pitch-encoder-checkpoint is required when "
                "--pitch-encoder=neutone-amt"
            )
        pitch_encoder_checkpoint = Path(pitch_encoder_checkpoint)
        if pitch_encoder_checkpoint.suffix.lower() == ".data":
            raise SystemExit(
                "--pitch-encoder-checkpoint must point to the .onnx file, "
                "not its .onnx.data weights"
            )
        if not pitch_encoder_checkpoint.is_file():
            raise SystemExit(
                f"pitch encoder checkpoint does not exist: {pitch_encoder_checkpoint}"
            )

        from polyddsp.preprocess import (
            NEUTONE_NATIVE_SR,
            audio_within_duration_limit,
            load_neutone_amt_model,
            neutone_amt_to_voices,
        )

        amt_model = load_neutone_amt_model(pitch_encoder_checkpoint, device)
        native_sr = int(getattr(
            amt_model.spec, "sample_rate", getattr(amt_model.spec, "sr", NEUTONE_NATIVE_SR),
        ))
        if not audio_within_duration_limit(
            Path(audio_path),
            max_samples=getattr(amt_model, "max_input_samples", None),
            sample_rate=native_sr,
        ):
            raise SystemExit("input exceeds the selected Neutone model's duration limit")
        pitch, velocity = neutone_amt_to_voices(
            Path(audio_path),
            sample_rate=int(cfg.model.sr),
            target_hop=int(cfg.model.frame_hop),
            n_voices=int(cfg.experiment.model.n_voices),
            device=device,
            model=amt_model,
            target_frames=target_frames,
        )
    else:
        raise SystemExit(
            f"unknown pitch encoder {pitch_encoder!r}; expected one of {_PITCH_ENCODERS}"
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
    pitch_encoder: str = "basic-pitch",
    pitch_encoder_checkpoint: str | Path | None = None,
) -> None:
    """Resynthesise one audio file through a checkpoint and write the result.

    The input is loaded once for the synthesis model and once at the selected
    pitch model's native rate, mirroring preprocessing. Output is peak-normalised,
    since DDSP output level is not calibrated to the input.
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

    # Always transcribe this input live. Explicit features override the
    # checkpoint's pitch encoder source; no pitch-cache sidecar is consulted.
    feats = pitch_features(
        audio_path,
        cfg,
        n_frames,
        device,
        pitch_encoder=pitch_encoder,
        pitch_encoder_checkpoint=pitch_encoder_checkpoint,
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
        f"pitch_encoder={pitch_encoder}, device={device}) -> {out_path}"
    )


def main(argv: list[str] | None = None) -> None:
    import sys

    p = argparse.ArgumentParser(
        prog="polyddsp-infer",
        description="Resynthesise an audio file through a trained PolyDDSP checkpoint.",
    )
    p.add_argument("--ckpt", required=True, help="path to best.pt / last.pt / top_*.pt")
    p.add_argument("--input", required=True, help="input audio file (any soundfile format)")
    p.add_argument("--out", required=True, help="output .wav path")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument(
        "--pitch-encoder", choices=_PITCH_ENCODERS, default="basic-pitch",
        help="live pitch encoder (default: basic-pitch)",
    )
    p.add_argument(
        "--pitch-encoder-checkpoint", default=None,
        help="Neutone Lightning checkpoint or ONNX export (required for neutone-amt)",
    )
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
        pitch_encoder=args.pitch_encoder,
        pitch_encoder_checkpoint=args.pitch_encoder_checkpoint,
    )


if __name__ == "__main__":
    main()
