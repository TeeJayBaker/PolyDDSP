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
- Loudness uses a trailing 512-sample Hann window (32 ms at 16 kHz), A-weighted
  power, and the fixed `(db / 80) + 1` scale. Like timbre features, each value
  ends at its 4 ms control-block boundary and uses zero-padding only for missing
  startup history. It requires no future audio. This changes conditioning from
  the old centered extractor and should be reflected in training/fine-tuning.
  The loudness metric also reuses this extractor, so historical scores used a
  different frame alignment.
- The timbre encoder uses 30 MFCC coefficients from 128 mel bands and per-frame
  LayerNorm. At 16 kHz its trailing 2048-sample Hann window spans 128 ms, with
  a 4 ms hop. Each frame ends at the last sample of its control block, uses
  zero-padding for missing startup history, and needs no future audio beyond
  that block. A partial final block is omitted. Frame timing stays fixed when
  audio is appended. The middle weighting emphasizes older audio; this is not
  a fixed 64 ms delay of the resulting embedding, nor a total latency guarantee.
  Full audio-to-audio resynthesis still needs an incremental input adapter. The changed
  normalization and alignment require fine-tuning or retraining z-enabled
  models; older analysis-buffer shapes may also need checkpoint migration.
- Audio can be generated at any sample rate by changing the synthesis
  resolution. The experiments below ran at 16 kHz to match the original DDSP.

Key modules: `polyddsp/model/polyddsp.py` (top level), `decoder.py`,
`additive.py`, `noise.py`, `reverb.py`, `z.py`, `loudness.py`,
`note_extraction.py`, `voice_allocation.py`.

The timbre encoder supports incremental inference with explicit per-stream state:

```python
encoder.eval()
state = None
for audio_chunk in incoming_audio:  # (batch, samples), at the encoder's sample rate
    z_chunk, state = encoder.stream(audio_chunk, state)
```

Each call returns only newly completed control frames. State retains a fixed-size
audio history, any partial hop, and the GRU's hidden state; previous feature frames
are not fed through the GRU again. Calls may have arbitrary sample counts, including
zero, while keeping the batch size, device, and dtype fixed. Passing `None` starts
a fresh stream. Discarding state at the end drops the partial final hop, matching
`encoder(audio)` on the concatenated input. `stream` disables autograd; use the
regular forward method for training. This method encodes timbre only; the
controls-to-audio streaming core below connects the synthesis components.

Loudness has the same streaming convention and needs only audio-buffer state:

```python
loudness_state = z_state = None
for audio_chunk in incoming_audio:
    loudness_chunk, loudness_state = model.loudness.stream(audio_chunk, loudness_state)
    z_chunk, z_state = model.z_encoder.stream(audio_chunk, z_state)  # use_z=True
```

With the same audio chunks and frame hop, both emit the same number of frames on
each call, even with irregular chunk sizes. Their windows have different history
lengths but end on identical sample boundaries. Setting `normalise=False` on the
loudness extractor returns raw dB; its streaming method disables autograd, while
the regular forward method remains differentiable.

Filtered noise and reverb also have explicit-state inference interfaces:

```python
noise_chunk, noise_state = model.noise_synth.stream(new_noise_mags, noise_state)
wet_chunk, reverb_state = model.reverb.stream(new_dry_audio, reverb_state)
```

Initialize each state to `None`; each call returns new audio and the updated state.
Noise accepts `(batch_times_voices, frames, noise_bands)` controls and retains FIR
overlap plus its own random-generator state. An optional `seed` on the first call
gives repeatable excitation independent of chunk boundaries. Supplying `noise=`
to either `forward` or `stream` bypasses random draws for exact comparisons.
The existing delay compensation is preserved: with 65 bands the stream withholds
62 samples (3.875 ms at 16 kHz) initially. Call `noise_synth.flush(noise_state)`
at EOF to emit the outstanding samples and match the offline output length.
`include_tail=True` also preserves the remaining physical FIR decay.

Reverb uses uniform partitioned FFT convolution, with a default 64-sample block
(4 ms at 16 kHz). It caches the partitioned IR and retains input-spectrum history,
overlap, and any incomplete input block. Its state size depends on the fixed IR
length, not performance duration. IR weights must remain fixed while the state is
in use; reset state after changing weights. `reverb.flush(reverb_state)` finishes
the pending block to match the offline input-length output. Use
`reverb.flush(reverb_state, include_tail=True)` to retain the complete reverb decay.
Discard state after flushing. Both streaming methods disable autograd; clip-based
forward methods remain differentiable and use the same synthesis weights.

The global streaming pipeline must account for unequal chunk lengths: noise's
startup compensation and reverb's partial-block buffering can return fewer samples
than were supplied. At EOF, flush upstream noise before flushing downstream reverb.
`StreamingSynthesizer` coordinates these buffers and output lengths for synthesis.

### Source-independent streaming synthesis

Pitch/velocity generation is separate from the synthesis core. Training can use
offline caches, an instrument adapter can generate controls from MIDI, and a future
live AMT adapter can supply the same frame format. No AMT inference is performed
by the streaming synthesizer:

```python
from polyddsp.model.streaming import StreamingSynthesizer

synth = StreamingSynthesizer.from_model(model).eval()
state = None
for pitch, velocity, loudness, z in control_chunks:
    audio, state = synth.process(pitch, velocity, loudness, z, state)
    # Consume ready audio in order; startup may produce an empty chunk.
tail = synth.flush(state, include_tail=True)
```

| Control | Shape | Meaning |
|---|---|---|
| pitch | `(batch, voices, frames)` | Hz; zero for an inactive slot |
| velocity | `(batch, voices, frames)` | Normalized note strength |
| loudness | `(batch, frames)` | Extractor scale `(db / 80) + 1` |
| z | `(batch, frames, z_dim)` | Explicit timbre conditioning for z-enabled models |

All controls share the same frame grid, device, and dtype. Voice slots persist
between calls; note tracking/allocation belongs to the input adapter. MIDI playback
must explicitly provide a loudness policy and, for z-enabled models, reference or
user-selected timbre conditioning. These are not inferred from pitch/velocity alone.

The core owns only the voice decoder and DSP modules, reusing the trained weights
and saved inference scales. Its state retains the GRU memory, oscillator phase,
at most two decoded context frames, small harmonic/noise alignment queues, and
noise/reverb state. Harmonic interpolation needs one future control frame (4 ms
at 16 kHz/hop 64); noise compensation and reverb block buffering are coordinated
internally. This is algorithmic buffering, not a measured device-to-device latency.
`None` resets a stream. Empty control blocks are accepted; batch size, voice count,
weights, scales, and timing settings must remain fixed until reset. `flush` ends
the stream and completes the input-length output; `include_tail=True` additionally
preserves the reverb decay. Discard state afterwards.

For differentiable clip training or evaluation with already prepared controls,
`model.synthesize(pitch, velocity, loudness, z)` returns `(audio, parts)` without
calling a pitch or audio encoder. The regular training forward computes loudness
and z from target audio and uses this same synthesis path with cached pitch and
velocity. `StreamingSynthesizer.process` is for inference and disables autograd.
The existing file-oriented `render` remains available. Live AMT, MIDI/audio-device
I/O, trained sound-quality evaluation, and the 32 ms latency target are separate
integration and validation work.

Benchmark a trained checkpoint's synthesis core with externally supplied synthetic
controls, without running a pitch model:

```bash
uv run python -m polyddsp.benchmark_stream --ckpt outputs/<run>/best.pt \
    --device cpu --threads 1 --block-frames 1
```

Repeat with `--block-frames 4` for 16 ms blocks at the default hop, or select
`--device cuda:0`. The report includes median, p99, maximum call times, and misses
of the block's processing deadline. CUDA measurements synchronize device execution.
This measures synthesis throughput, not sound quality or total audio-device latency.
Choose the training architecture using these measurements and subsequent trained
quality evaluation; a functionally streaming model may still be too slow for its
selected block size on a particular device.

## Install

```bash
uv sync
export POLYDDSP_DATA_DIR=/path/to/data      # datasets, default /data/polyddsp
export POLYDDSP_CACHE_DIR=/path/to/cache    # cache base, default ./data
export POLYDDSP_OUT_DIR=/path/to/runs       # run outputs, default outputs/
```

## Data

| Dataset | Download | Path under `$POLYDDSP_DATA_DIR` |
|---|---|---|
| GuitarSet | [zenodo 3371780](https://zenodo.org/records/3371780) (`audio_mono-pickup_mix.zip`) | `guitarset/audio_mono-pickup_mix/*_mix.wav` |
| MAESTRO | [magenta.tensorflow.org](https://magenta.tensorflow.org/datasets/maestro) | `MAESTRO/maestro-v3.0.0/**/*.wav` |

Files are split 80/20 train/val from `run.seed`.

## Usage

Machine-local paths, GPU choices, and the selected pitch-cache source live in
`configs/local/default.yaml`, which is ignored by Git and loaded automatically.
Start from the committed template:

```bash
cp configs/local/example.yaml configs/local/default.yaml
```

Edit that local file once, then use the committed training profiles:

```bash
polyddsp-train train_profile=guitarset
polyddsp-train train_profile=gaps
polyddsp-train train_profile=maestro
```

The profiles in `configs/train_profile/` contain portable experiment and
training settings; they do not select a pitch-cache source. Normal Hydra
overrides still work, for example
`polyddsp-train train_profile=maestro train.steps=50000`.

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

Training uses one device. It defaults to `device=auto`, which selects CUDA when
available and CPU otherwise. Select a GPU explicitly with, for example,
`polyddsp-train experiment=guitarset device=cuda:1`; use `device=cpu` to force CPU.
As usual, CUDA indices refer to the devices visible through `CUDA_VISIBLE_DEVICES`.

The experimental warmup can gate z conditioning and synthesized noise
independently. Noise warmup follows the z warmup timing by default, so this
enables both with a 40% blocked phase and a 10% ramp:

```bash
polyddsp-train train_profile=guitarset train.z_warmup.enabled=true
```

Override `train.noise_warmup.enabled`, `block_pct`, or `ramp_pct` to separate
the schedules. `noise_scale=0` is applied to the final noise waveform, giving
the noise branch zero output and zero reconstruction gradient. TensorBoard logs
both schedules as `train/z_scale` and `train/noise_scale`.

Each checkpoint stores the z and noise scales used when it was evaluated.
Inference and standalone evaluation reuse those scales, so a `best.pt` selected
during the ramp reproduces that checkpoint rather than silently enabling both
paths fully. Use `--z-scale` and `--noise-scale` with `polyddsp-infer` to override
the saved values deliberately. Older checkpoints recover their scales from the
saved step and the run's adjacent `config.yaml`.

Inference uses Basic Pitch by default. To transcribe with Neutone AMT instead,
select it and provide the same kind of Lightning checkpoint or ONNX export used
by preprocessing:

```bash
polyddsp-infer --ckpt outputs/<run>/best.pt --input clip.wav --out resynth.wav \
    --pitch-encoder neutone-amt-nonstreaming \
    --pitch-encoder-checkpoint ./checkpoints/NeutoneAMT/amt.onnx
```

`basic-pitch`, `neutone-amt-streaming`, and `neutone-amt-nonstreaming` all transcribe in-process and do
not read or write a pitch-cache sidecar. The live transcription is passed
directly to PolyDDSP regardless of the checkpoint's configured pitch source.

Pitch caches use one layout everywhere:

```text
data/<pitch-cache-source>/<experiment>/...
```

The four independent source namespaces are `basic-pitch`,
`neutone-amt-streaming`, `neutone-amt-nonstreaming`, and `ground-truth`.
Preprocessing derives its output directory from `model` and the experiment;
training derives the same directory from `run.pitch_cache_source`. Relative
paths beneath the dataset root are preserved inside the experiment directory.
Set `run.cache_root=/another/cache/root` to move the whole cache tree. Only set
`output_dir=null` when intentionally writing legacy sidecars beside source
audio.

To preprocess with Neutone AMT, pass a Lightning checkpoint or an ONNX export:

```bash
polyddsp-preprocess experiment=guitarset \
    model=neutone-amt-nonstreaming \
    model_path=./checkpoints/NeutoneAMT/amt.onnx
```

This writes to `./data/neutone-amt-nonstreaming/guitarset` by default. The
streaming model instead writes to `./data/neutone-amt-streaming/guitarset`, so
the two variants never share a cache directory.

Keep any accompanying `amt.onnx.data` file beside `amt.onnx`. Both streaming
exports (audio plus state cache) and offline exports are supported. Select them
with `model=neutone-amt-streaming` and `model=neutone-amt-nonstreaming`,
respectively; preprocessing verifies the selection against the ONNX graph. Streaming
state resets for each file; multi-delay exports use the first/lowest-delay
readout, matching the checkpoint backend. Timing comes from export metadata,
with 44.1 kHz / hop 512 defaults for older exports. Output caches use the
`neutone_streaming_v{V}_sr{sample_rate}_hop{hop}` or
`neutone_nonstreaming_v{V}_sr{sample_rate}_hop{hop}` suffix and constant MIDI velocity 64.
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
polyddsp-train experiment=guitarset run.pitch_cache_source=neutone-amt-nonstreaming
```

Pass the same source override when evaluating. `run.pitch_cache_source` defaults
to `basic-pitch`; selecting `neutone-amt-nonstreaming` uses the
`neutone_nonstreaming_v6_sr16000_hop64` suffix for this configuration. All three
options use the existing `cached_basic_pitch` model input, which reads the shared
pitch/velocity tensor format. New caches also record their source internally;
training rejects a cache whose recorded source disagrees with
`pitch_cache_source`, while legacy caches without metadata remain readable.

Ground-truth caches use `run.pitch_cache_source=ground-truth` and the suffix
`ground_truth_v{V}_sr{sample_rate}_hop{hop}`. Generate them with:

```bash
polyddsp-preprocess experiment=maestro model=ground-truth
```

Each audio file is paired with the unique MIDI file having the same filename
stem among files selected by `midi_glob`. Both globs are evaluated beneath the
same dataset root, but independently: `file_glob` selects audio and `midi_glob`
selects MIDI. For GAPS, point `data_root` at the GAPS directory and use globs
such as `audio/*wc.wav` and `midi/*.mid`.
Ground-truth
preprocessing applies sustain pedal, normalized note velocity, and track pitch
bends (default full-scale range: `pitch_bend_range=2.0` semitones). Disable pedal
extension with `apply_sustain=false` when the MIDI note durations already include
it.

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

- `configs/preprocess.yaml` owns preprocessing settings. `data_root`,
  `file_glob`, `midi_glob`, and `n_voices` default to values from the
  selected experiment; each can also be overridden directly. Select the pitch
  model with `model=basic-pitch`, `model=neutone-amt-streaming`,
  `model=neutone-amt-nonstreaming`, or `model=ground-truth`.
  Set `sample_rate` and `hop` to match the training model's
  `model.sr` and `model.frame_hop`, and match the voice count too. Training's
  `run.pitch_cache_source` selects which generated cache to read (one of those
  same four names). Training and preprocessing both resolve
  `run.cache_root/<source>/<experiment>` automatically.
- Each cache is a per-file `.f0.pt` tensor file whose suffix identifies its
  source, voice count, sample rate, and hop. It is skipped if newer than its
  audio (and MIDI, for ground truth).
- `infer` transcribes in-process with `--pitch-encoder basic-pitch` (the default)
  or either Neutone AMT variant, so it needs no cache and takes any audio file.
- Config is Hydra (`configs/config.yaml`, `configs/preprocess.yaml`, and
  `configs/experiment/*.yaml`); any key can be overridden on the CLI, e.g.
  `train.batch_size=8`. Checkpoints, the resolved config, `log.jsonl` and metric
  summaries go to `$POLYDDSP_OUT_DIR/<run.name>/` (default: `outputs/<run.name>/`).
  Default run names use `experiment_YYMMDDTHHmm_pitch-cache-source`, for example
  `guitarset_260930T0915_neutone-amt-nonstreaming`.

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
- `train/z_scale` and `train/noise_scale` are overlaid under
  Custom Scalars → Training → Warmup scales.
- Harmonic and noise RMS are stored in the main run as
  `val/component_rms/{harmonic,noise}` and overlaid under
  Custom Scalars → Validation → Component RMS.
- Audio retains all six previews: reference, prediction, harmonic, noise,
  peak-normalized dry mix, and wet-only reverb. Periodic and final-model previews
  are labelled `val/last.pt/audio_*`; the best validation-MSS checkpoint is
  rendered at shutdown under `val/best.pt/audio_*`.
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
- The file-oriented renderer remains offline. `StreamingSynthesizer` instead
  accepts externally prepared controls and carries decoder, oscillator, noise,
  and reverb state. Live pitch transcription/voice allocation and MIDI/audio-device
  adapters are separate work. Real-time suitability depends on the model size,
  block size, and target device and must be benchmarked; offline throughput does
  not establish the 32 ms total latency target.

## Tests

```bash
uv run pytest
```

The CLAP and FAD metric tests download model checkpoints on first run.

## License

Code is MIT (see `LICENSE`). The paper is CC BY 4.0.
