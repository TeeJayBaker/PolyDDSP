# PolyDDSP

A PyTorch reimplementation of PolyDDSP — a polyphonic extension of
[DDSP](https://arxiv.org/abs/2001.04643) that resynthesises polyphonic audio
through a bank of differentiable harmonic + filtered-noise voices.

Paper: Baker, T., Climent, R., & Chen, K. *PolyDDSP: A Lightweight and
Polyphonic Differentiable Digital Signal Processing Library.* Music in the AI
Era (CMMR 2023), Springer LNCS 13770 — https://zenodo.org/records/10113134

## Architecture

- Pitch extraction is a frozen Basic Pitch, using the full pre-mean per-frame
  note amplitudes and pitch bend values, with loudest-voice allocation.
- All voices share a single set of decoder weights and are batch processed, so
  voice count is a runtime choice.
- Each voice takes its loudness from one extracted loudness envelope plus its
  own framed velocity from Basic Pitch. Timbre encoding is global.
- Audio can be generated at any sample rate by changing the synthesis
  resolution. The experiments below ran at 16 kHz to match the original DDSP.

Key modules: `polyddsp/model/polyddsp.py` (top level), `decoder.py`,
`additive.py`, `noise.py`, `reverb.py`, `z.py`, `loudness.py`,
`note_extraction.py`, `voice_allocation.py`.

## Install

```bash
uv sync
export POLYDDSP_DATA_DIR=/path/to/data      # datasets, default /data/polyddsp
export POLYDDSP_OUT_DIR=/path/to/runs       # run outputs, default outputs/
```

## Data

| Dataset | Download | Path under `$POLYDDSP_DATA_DIR` |
|---|---|---|
| GuitarSet | [zenodo 3371780](https://zenodo.org/records/3371780) (`audio_mono-pickup_mix.zip`) | `guitarset/**/*_mix.wav` |
| MAESTRO | [magenta.tensorflow.org](https://magenta.tensorflow.org/datasets/maestro) | `maestro/bach_2018/**/*.wav` |

MAESTRO is restricted to `bach_2018` to keep the piano and recording
environment consistent. Files are split 80/20 train/val from `run.seed`.

## Usage

```bash
# 1. Transcribe the dataset to a pitch cache. Required before training.
polyddsp-preprocess --n-voices 6 \
    --glob '**/*_mix.wav' --root $POLYDDSP_DATA_DIR/guitarset

# 2. Train
polyddsp-train experiment=guitarset

# 3. Evaluate a checkpoint
python -m polyddsp.eval experiment=guitarset ckpt=outputs/<run>/best.pt

# 4. Render audio from a checkpoint
polyddsp-infer ckpt=outputs/<run>/best.pt input=clip.wav out=resynth.wav
```

By default preprocessing writes each cache beside its source audio. To keep the
dataset untouched, add `--output-dir preprocessed/guitarset`, then train with
`experiment.dataset.pitch_cache_root=preprocessed/guitarset`. Nested paths are
preserved beneath the separate cache directory.

- `--n-voices` must match the config's `model.n_voices`. `--glob` defaults to
  `**/*.mp3`, so pass it explicitly; for MAESTRO use
  `--glob '**/*.wav' --root $POLYDDSP_DATA_DIR/maestro/bach_2018`.
- The cache is a per-file `.bp_v{V}_sr16000_hop64.f0.pt` sidecar, and is skipped
  if newer than its audio.
- `infer` transcribes in-process, so it needs no cache and takes any audio file.
- Config is Hydra (`configs/config.yaml` + `configs/experiment/*.yaml`); any key
  can be overridden on the CLI, e.g. `train.batch_size=8`. Checkpoints, the
  resolved config, `log.jsonl` and metric summaries go to
  `$POLYDDSP_OUT_DIR/<run.name>/`. Pass `wandb.mode=disabled` to run offline.

## Results

Held-out val split, 100k steps at batch size 12 on 4-second crops. All metrics
lower-is-better; machine-readable copy in `results/final_metrics.json`.

| | GuitarSet | MAESTRO |
|---|---|---|
| `n_voices` | 6 | 6 |
| `use_z` | true | false |
| `use_reverb` | false | true |
| `use_noise` | true | false |
| MSS | 0.642 | 0.423 |
| CLAP-FD | 0.0495 | 0.0523 |
| FAD | 0.438 | 1.235 |
| multipitch MSE | 0.000956 | 0.00117 |
| MFCC L1 | 1.607 | 1.675 |
| loudness L1 | 1.780 | 2.408 |

```bash
polyddsp-train experiment=guitarset
polyddsp-train experiment=maestro experiment.model.n_voices=6
```

MAESTRO ships `n_voices: 10`; the run above used 6, which fits in 24 GB and was
never exceeded by simultaneous notes in this subset.

## Performance

Faster than realtime on CPU, scaling with voice count: roughly 15x at V=1, 3.4x
at V=6 and 2x at V=10.

## Known limitations

- Basic Pitch note timing drifts on long files, by about 0.3 s per minute of
  audio. See `tests/test_bp_timing_drift.py`.
- The model is not streamable, purely because there is no streamable polyphonic
  pitch detector to put in front of it. Everything downstream generalises to
  streaming.

## Tests

```bash
uv run pytest
```

The CLAP and FAD metric tests download model checkpoints on first run.

## License

Code is MIT (see `LICENSE`). The paper is CC BY 4.0.
