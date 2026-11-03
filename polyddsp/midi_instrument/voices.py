"""Causal note allocation shared by reference rendering and model controls."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyddsp.midi_instrument.events import MidiEvent
from polyddsp.midi_instrument.renderer import ReferenceRenderer


@dataclass
class _Voice:
    channel: int
    note: int
    velocity: int
    order: int
    held: bool = True


class VoiceAllocator:
    def __init__(self, renderer: ReferenceRenderer, *, pitch_bend_range: float = 2.0) -> None:
        if not 0 < pitch_bend_range < 128:
            raise ValueError("pitch_bend_range must be in (0, 128)")
        self.renderer = renderer
        self.voices: list[_Voice | None] = [None] * renderer.n_voices
        self.pitch_bend_range = pitch_bend_range
        self.sustain = [False] * 16
        self.bends = [0] * 16
        self.controllers = [{1: 0, 7: 100, 10: 64, 11: 127} for _ in range(16)]
        self._order = 0

    def _release(self, slot: int) -> None:
        voice = self.voices[slot]
        if voice is not None:
            self.renderer.note_off(slot, voice.note)
            self.voices[slot] = None

    def handle(self, event: MidiEvent) -> None:
        channel = event.channel
        if event.kind == "note_on" and event.velocity > 0:
            # Repeated note-ons retrigger one slot rather than creating ambiguous duplicate keys.
            slot = next((i for i, v in enumerate(self.voices)
                         if v is not None and (v.channel, v.note) == (channel, event.note)), None)
            if slot is None:
                slot = next((i for i, v in enumerate(self.voices) if v is None), None)
            if slot is None:
                slot = min(range(len(self.voices)), key=lambda i: (self.voices[i].velocity, self.voices[i].order))
            self.renderer.stop(slot)  # also removes an old release tail when reusing a free slot
            self._order += 1
            self.voices[slot] = _Voice(channel, event.note, event.velocity, self._order)
            self.renderer.pitch_bend(slot, self.bends[channel])
            for control, value in self.controllers[channel].items():
                self.renderer.control_change(slot, control, value)
            self.renderer.note_on(slot, event.note, event.velocity)
        elif event.kind in {"note_on", "note_off"}:
            for slot, voice in enumerate(self.voices):
                if voice is not None and (voice.channel, voice.note) == (channel, event.note):
                    voice.held = False
                    if not self.sustain[channel]:
                        self._release(slot)
        elif event.kind == "pitchwheel":
            self.bends[channel] = event.pitch
            for slot, voice in enumerate(self.voices):
                if voice is not None and voice.channel == channel:
                    self.renderer.pitch_bend(slot, event.pitch)
        elif event.kind == "control_change":
            control, value = event.control, event.value
            if control == 64:
                self.sustain[channel] = value >= 64
                for slot, voice in enumerate(self.voices):
                    if voice is not None and voice.channel == channel and not voice.held and value < 64:
                        self._release(slot)
            elif control in {120, 123}:
                for slot, voice in enumerate(self.voices):
                    if voice is not None and voice.channel == channel:
                        voice.held = False
                        if control == 120:
                            self.renderer.stop(slot)
                            self.voices[slot] = None
                        elif not self.sustain[channel]:
                            self._release(slot)
            elif control in self.controllers[channel]:
                self.controllers[channel][control] = value
                for slot, voice in enumerate(self.voices):
                    if voice is not None and voice.channel == channel:
                        self.renderer.control_change(slot, control, value)
            # Bank selection, program changes and RPN changes do not alter the fixed patch/range.

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        pitch = np.zeros(len(self.voices), dtype=np.float32)
        velocity = np.zeros_like(pitch)
        for slot, voice in enumerate(self.voices):
            if voice is not None:
                bend = self.bends[voice.channel] * self.pitch_bend_range / 8192.0
                pitch[slot] = 440.0 * 2.0 ** ((voice.note + bend - 69.0) / 12.0)
                velocity[slot] = voice.velocity / 127.0
        return pitch, velocity

    def all_notes_off(self) -> None:
        """Release keys and pedal at EOF, retaining the renderer's natural decay."""
        self.sustain = [False] * 16
        for slot in range(len(self.voices)):
            self._release(slot)
