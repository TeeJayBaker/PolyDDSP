"""MIDI → muted reference audio → streaming features → PolyDDSP output."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch

from polyddsp.midi_instrument.events import MidiEvent
from polyddsp.midi_instrument.renderer import ReferenceRenderer
from polyddsp.midi_instrument.voices import VoiceAllocator
from polyddsp.model.streaming import StreamingSynthesizer

if TYPE_CHECKING:
    from polyddsp.model.polyddsp import PolyDDSP


class MidiInstrument:
    """Single-stream adapter; renderer and all model state belong to this instance.

    process advances an absolute model-sample clock. Events must lie in its
    half-open interval [sample_position, sample_position + samples). Rendering
    splits at event times and control boundaries. Snapshots are taken at block
    ends, before events at the next block's start, matching feature extraction.
    Reference audio is never included in the returned model output.
    """

    def __init__(
        self, model: PolyDDSP, renderer: ReferenceRenderer, *,
        pitch_bend_range: float = 2.0, output_gain: float = 1.0,
        noise_seed: int | None = None,
    ) -> None:
        if renderer.sample_rate != model.sr or renderer.n_voices != model.n_voices:
            raise ValueError("renderer sample rate and voice count must match the model")
        if output_gain < 0 or not np.isfinite(output_gain):
            raise ValueError("output_gain must be finite and nonnegative")
        self.model = model.eval()
        self.renderer = renderer
        self.allocator = VoiceAllocator(renderer, pitch_bend_range=pitch_bend_range)
        self.synth = StreamingSynthesizer.from_model(model, noise_seed=noise_seed).eval()
        self.sample_rate, self.frame_hop = model.sr, model.frame_hop
        self.device = next(model.decoder.parameters()).device
        self.dtype = next(model.decoder.parameters()).dtype
        self.output_gain = output_gain
        self.sample_position = 0
        self._loudness_state = self._z_state = self._synthesis_state = None
        self._finished = self._closed = False

    def _output(self, audio: torch.Tensor) -> np.ndarray:
        return (audio[0].detach().cpu().numpy() * self.output_gain).astype(np.float32, copy=False)

    @torch.no_grad()
    def process(self, samples: int, events: Iterable[MidiEvent] = ()) -> np.ndarray:
        if self._finished or self._closed:
            raise RuntimeError("instrument is finished or closed; create a new instance")
        if not isinstance(samples, int) or samples < 0:
            raise ValueError("samples must be a nonnegative integer")
        events = sorted(events, key=lambda event: event.sample)  # stable for simultaneous events
        end = self.sample_position + samples
        if any(not self.sample_position <= event.sample < end for event in events):
            raise ValueError("events must fall within this call's half-open sample interval")
        if samples == 0:
            return np.empty(0, dtype=np.float32)
        audio_parts, pitch_frames, velocity_frames = [], [], []
        event_index = 0
        while self.sample_position < end:
            while event_index < len(events) and events[event_index].sample == self.sample_position:
                self.allocator.handle(events[event_index])
                event_index += 1
            next_event = events[event_index].sample if event_index < len(events) else end
            next_frame = (self.sample_position // self.frame_hop + 1) * self.frame_hop
            stop = min(next_event, next_frame, end)
            segment = np.asarray(self.renderer.render(stop - self.sample_position), dtype=np.float32)
            if segment.shape != (stop - self.sample_position,) or not np.isfinite(segment).all():
                raise RuntimeError("reference renderer must return finite mono audio of the requested length")
            audio_parts.append(segment)
            self.sample_position = stop
            if stop % self.frame_hop == 0:
                pitch, velocity = self.allocator.snapshot()
                pitch_frames.append(pitch)
                velocity_frames.append(velocity)
        audio = torch.from_numpy(np.concatenate(audio_parts)).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        loudness, self._loudness_state = self.model.loudness.stream(audio, self._loudness_state)
        z = None
        if self.model.use_z:
            z, self._z_state = self.model.z_encoder.stream(audio, self._z_state)
        if pitch_frames:
            pitch = torch.from_numpy(np.stack(pitch_frames, axis=1)).to(device=self.device, dtype=self.dtype).unsqueeze(0)
            velocity = torch.from_numpy(np.stack(velocity_frames, axis=1)).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        else:
            pitch = audio.new_empty(1, self.model.n_voices, 0)
            velocity = torch.empty_like(pitch)
        if pitch.shape[-1] != loudness.shape[-1] or (z is not None and z.shape[1] != pitch.shape[-1]):
            raise RuntimeError("reference features and MIDI controls lost frame alignment")
        output, self._synthesis_state = self.synth.process(pitch, velocity, loudness, z, self._synthesis_state)
        return self._output(output)

    def release_all(self) -> None:
        if self._finished or self._closed:
            raise RuntimeError("instrument is finished or closed")
        self.allocator.all_notes_off()

    def finish(self, *, include_tail: bool = True) -> np.ndarray:
        """Pad at most one final partial hop, flush synthesis, and end the stream.

        Padding advances the actual reference renderer, preserving its decay.
        To hear note releases, call release_all and process a desired release
        interval before finish. include_tail preserves model reverb decay.
        """
        if self._finished or self._closed:
            raise RuntimeError("instrument is finished or closed")
        parts = []
        remainder = self.sample_position % self.frame_hop
        if remainder:
            parts.append(self.process(self.frame_hop - remainder))
        if self._synthesis_state is not None:
            parts.append(self._output(self.synth.flush(self._synthesis_state, include_tail=include_tail)))
        self._finished = True
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)

    def close(self) -> None:
        if not self._closed:
            self.renderer.close()
            self._closed = True

    def __enter__(self) -> MidiInstrument:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
