"""Headless FluidSynth rendering; the reference audio never opens an output device."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol

import numpy as np


class ReferenceRenderer(Protocol):
    sample_rate: int
    n_voices: int

    def note_on(self, slot: int, note: int, velocity: int) -> None: ...
    def note_off(self, slot: int, note: int) -> None: ...
    def stop(self, slot: int) -> None: ...
    def pitch_bend(self, slot: int, value: int) -> None: ...
    def control_change(self, slot: int, control: int, value: int) -> None: ...
    def render(self, samples: int) -> np.ndarray: ...
    def close(self) -> None: ...


class FluidSynthRenderer:
    """Render a fixed SF2 preset into normalized mono samples at the model rate.

    Each model slot has a dedicated melodic FluidSynth channel. Channel 9 is
    skipped to avoid the General MIDI percussion channel. SoundFont effects are
    disabled; gain is fixed, never normalized independently for each block.
    """

    def __init__(
        self, soundfont: str | Path, sample_rate: int, n_voices: int,
        *, bank: int = 0, preset: int = 24, gain: float = 0.2,
        pitch_bend_range: float = 2.0,
    ) -> None:
        soundfont = Path(soundfont)
        if not soundfont.is_file():
            raise FileNotFoundError(f"SoundFont not found: {soundfont}")
        if sample_rate < 8_000 or not 1 <= n_voices <= 255 or gain <= 0 or not np.isfinite(gain):
            raise ValueError("sample_rate must be >= 8000, n_voices in [1, 255], and gain finite and positive")
        if not 0 <= bank <= 16383 or not 0 <= preset <= 127:
            raise ValueError("bank must be in [0, 16383] and preset in [0, 127] (zero-based)")
        if not 0 < pitch_bend_range < 128:
            raise ValueError("pitch_bend_range must be in (0, 128) semitones")
        try:
            fluidsynth = import_module("fluidsynth")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "FluidSynth is unavailable. Install the native FluidSynth library and "
                "run `uv sync --extra midi`. The Python binding alone does not include the native library."
            ) from exc
        self.sample_rate, self.n_voices = sample_rate, n_voices
        self._channels = [slot if slot < 9 else slot + 1 for slot in range(n_voices)]
        channels = max(16, ((max(self._channels) + 16) // 16) * 16)
        self._synth = fluidsynth.Synth(
            samplerate=float(sample_rate), channels=channels, gain=float(gain),
            **{"synth.reverb.active": 0, "synth.chorus.active": 0},
        )
        try:
            sfid = self._synth.sfload(str(soundfont))
            if sfid < 0:
                raise RuntimeError(f"FluidSynth could not load SoundFont: {soundfont}")
            for channel in self._channels:
                if self._synth.program_select(channel, sfid, bank, preset) < 0:
                    raise RuntimeError(f"SoundFont has no selectable bank {bank}, preset {preset}")
                # RPN 0 configures bend sensitivity, keeping renderer and Hz controls consistent.
                for control, value in ((101, 0), (100, 0), (6, int(pitch_bend_range)),
                                       (38, round((pitch_bend_range % 1) * 100)), (101, 127), (100, 127)):
                    self._synth.cc(channel, control, value)
        except Exception:
            self.close()
            raise
        # Do NOT call Synth.start(): only get_samples() is used.

    def _channel(self, slot: int) -> int:
        if not 0 <= slot < self.n_voices:
            raise ValueError("voice slot is outside the configured renderer")
        if self._synth is None:
            raise RuntimeError("renderer is closed")
        return self._channels[slot]

    def note_on(self, slot: int, note: int, velocity: int) -> None:
        channel = self._channel(slot)
        self._synth.noteon(channel, note, velocity)

    def note_off(self, slot: int, note: int) -> None:
        channel = self._channel(slot)
        self._synth.noteoff(channel, note)

    def stop(self, slot: int) -> None:
        channel = self._channel(slot)
        self._synth.cc(channel, 120, 0)

    def pitch_bend(self, slot: int, value: int) -> None:
        channel = self._channel(slot)
        self._synth.pitch_bend(channel, value)

    def control_change(self, slot: int, control: int, value: int) -> None:
        channel = self._channel(slot)
        self._synth.cc(channel, control, value)

    def render(self, samples: int) -> np.ndarray:
        if samples < 0:
            raise ValueError("samples must be nonnegative")
        if self._synth is None:
            raise RuntimeError("renderer is closed")
        if samples == 0:
            return np.empty(0, dtype=np.float32)
        stereo = np.asarray(self._synth.get_samples(samples))
        if stereo.size != samples * 2 or stereo.dtype != np.int16:
            raise RuntimeError("FluidSynth returned unexpected interleaved int16 stereo audio")
        return stereo.reshape(samples, 2).astype(np.float32).mean(axis=1) / 32768.0

    def close(self) -> None:
        if self._synth is not None:
            self._synth.delete()
            self._synth = None
