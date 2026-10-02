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
polyddsp-preprocess experiment=guitarset

# 2. Train
polyddsp-train experiment=guitarset

# 3. Evaluate a checkpoint
python -m polyddsp.eval experiment=guitarset ckpt=outputs/<run>/best.pt

# 4. Render audio from a checkpoint
polyddsp-infer --ckpt outputs/<run>/best.pt --input clip.wav --out resynth.wav
```

Inference uses Basic Pitch by default. To transcribe with Neutone AMT instead,
select it and provide the same kind of Lightning checkpoint or ONNX export used
by preprocessing:

```bash
polyddsp-infer --ckpt outputs/<run>/best.pt --input clip.wav --out resynth.wav \
    --pitch-encoder neutone-amt \
    --pitch-encoder-checkpoint ./checkpoints/NeutoneAMT/amt.onnx
```

`basic-pitch` and `neutone-amt` both transcribe in-process and do
not read or write a pitch-cache sidecar. The live transcription is passed
directly to PolyDDSP regardless of the checkpoint's configured pitch source.

By default preprocessing writes each cache beside its source audio. To keep the
dataset untouched, add `output_dir=preprocessed/guitarset`. When
training or evaluating, read those caches with
`experiment.dataset.pitch_cache_root=preprocessed/guitarset`. Nested paths are
preserved beneath the separate output directory. `output_dir=null`
keeps the default of writing beside the source audio.

To preprocess with Neutone AMT, pass a Lightning checkpoint or an ONNX export:

```bash
polyddsp-preprocess experiment=guitarset \
    data_root=/media/ssd2/data/guitarset/audio_mono-pickup_mix \
    output_dir=./data/neutone-amt/guitarset \
    model=neutone-amt \
    model_path=./checkpoints/NeutoneAMT/amt.onnx
```

Keep any accompanying `amt.onnx.data` file beside `amt.onnx`. Both streaming
exports (audio plus state cache) and offline exports are supported. Streaming
state resets for each file; multi-delay exports use the first/lowest-delay
readout, matching the checkpoint backend. Timing comes from export metadata,
with 44.1 kHz / hop 512 defaults for older exports. Output caches use the
`neutone_v{V}_sr{sample_rate}_hop{hop}` suffix and constant MIDI velocity 64.
ONNX Runtime is installed by `uv sync`, with CPU and CUDA support on Linux
x86-64 and CPU support on other platforms. Use `device=cpu` to force
CPU inference or `device=cuda:0` to select a GPU.

For nonstreaming ONNX on CUDA, preprocessing checks audio headers and skips files
longer than 65,000 model hops (about 12 minutes 35 seconds at 44.1 kHz / hop 512).
This leaves room for internal padding below cuDNN's 65,535-step RNN limit.
Each skip prints the filename, duration, and limit without loading the waveform
or writing a cache. Streaming ONNX and CPU inference do not use this limit.
Skipped recordings must be excluded from the training input or preprocessed
separately with the streaming export or CPU.

To train using these Neutone caches:

```bash
polyddsp-train experiment=guitarset \
    experiment.dataset.root=/media/ssd2/data/guitarset/audio_mono-pickup_mix \
    experiment.dataset.pitch_cache_root=./data/neutone-amt/guitarset \
    experiment.dataset.pitch_cache_source=neutone-amt
```

Pass the same dataset overrides when evaluating. `pitch_cache_source` defaults
to `basic-pitch`; selecting `neutone-amt` uses the `neutone_v6_sr16000_hop64` suffix
for this configuration. Both use the existing `cached_basic_pitch` pitch source,
which reads the shared pitch/velocity tensor format.

Set `parallel=true` to process files concurrently with
[Memex](https://github.com/bgenchel/Memex). Without it, processing is sequential.
Memex runs one file first to estimate memory needs, then chooses concurrency
from available memory while preserving `memory_headroom=2048` MiB per
device. With parallel processing, `device=cuda` lets Memex spread
tasks across all GPUs exposed by `CUDA_VISIBLE_DEVICES`;
`device=cuda:0` pins work to the first visible GPU, and
`device=cpu` uses CPU processes.

Each task loads its own model and processes up to
`files_per_task=4` files sequentially before exiting. This setting
applies only to parallel preprocessing; it is neither the number of workers nor
an inference batch size. Increase that setting to
amortize startup on short clips, or lower it for more frequent progress updates
and finer scheduling. Blocks within a streaming recording remain sequential.
Existing fresh caches are skipped before starting workers, and completed cache
files are published atomically so Memex can retry an out-of-memory task safely.
More workers use more memory; throughput depends on the model and available
compute.

- `configs/preprocess.yaml` owns preprocessing settings. Only `data_root`,
  `file_glob`, and `n_voices` default to values from the
  selected experiment; each can also be overridden directly. Select the pitch
  model with `model=basic-pitch` or `model=neutone-amt`.
  Set `sample_rate` and `hop` to match the training model's
  `model.sr` and `model.frame_hop`, and match the voice count too. Training's
  `experiment.dataset.pitch_cache_source` selects which generated cache to
  read (`basic-pitch` or `neutone-amt`); its `pitch_cache_root` points to the
  preprocessing output directory when using separate cache storage.
- The cache is a per-file `.bp_v{V}_sr16000_hop64.f0.pt` sidecar, and is skipped
  if newer than its audio.
- `infer` transcribes in-process with `--pitch-encoder basic-pitch` (the default)
  or `--pitch-encoder neutone-amt`, so it needs no cache and takes any audio file.
- Config is Hydra (`configs/config.yaml`, `configs/preprocess.yaml`, and
  `configs/experiment/*.yaml`); any key can be overridden on the CLI, e.g.
  `train.batch_size=8`. Checkpoints, the resolved config, `log.jsonl` and metric
  summaries go to `$POLYDDSP_OUT_DIR/<run.name>/` (default: `outputs/<run.name>/`).

### Training logs

Training logs locally to TensorBoard; no account or logging service is required.
Start the dashboard with:

```bash
uv run tensorboard --logdir "${POLYDDSP_OUT_DIR:-outputs}"
```

Open `http://localhost:6006`. Each run writes events under
`<run.out_dir>/tensorboard/` (or the reused run directory when resuming):

- Scalars retain the existing tags: `loss`, `val/<metric>`, and
  `val/final/<metric>`, at the same training/evaluation intervals.
- Audio retains all six previews: reference, prediction, harmonic, noise,
  peak-normalized dry mix, and wet-only reverb (`val/audio_*`).
- The resolved Hydra config appears in the Text tab under `config`.

Resuming reuses the run's event directory and hides events at or beyond the
checkpoint step before writing replacement data. `log.jsonl` remains append-only.
Set `tensorboard.enabled=false` to disable TensorBoard events and audio previews
while keeping JSONL metrics and checkpoints. `tensorboard.flush_secs=30` controls
the background flush interval; closing the logger flushes pending events.

The old `wandb.*` overrides are removed; drop `wandb.mode=online` from existing
commands, or replace `wandb.mode=disabled` with `tensorboard.enabled=false`.
Historical W&B runs are not converted. Hosted sharing, W&B's automatic system
telemetry, and its config-comparison UI are not reproduced by this local logger.
The W&B package is still an indirect dependency of the CLAP metrics package,
but PolyDDSP no longer initializes a W&B run.

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
