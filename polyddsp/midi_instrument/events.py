"""MIDI events timestamped in the model's absolute sample clock."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MidiEvent:
    sample: int
    kind: str
    channel: int = 0
    note: int = 0
    velocity: int = 0
    control: int = 0
    value: int = 0
    pitch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sample, int) or self.sample < 0:
            raise ValueError("event sample must be a nonnegative integer")
        if self.kind not in {"note_on", "note_off", "control_change", "pitchwheel"}:
            raise ValueError(f"unsupported MIDI event: {self.kind}")
        if not 0 <= self.channel <= 15:
            raise ValueError("MIDI channel must be in [0, 15]")
        if any(not 0 <= item <= 127 for item in (self.note, self.velocity, self.control, self.value)):
            raise ValueError("MIDI note, velocity, controller and value must be in [0, 127]")
        if not -8192 <= self.pitch <= 8191:
            raise ValueError("pitch bend must be in [-8192, 8191]")

    @classmethod
    def from_message(cls, message, sample: int) -> MidiEvent | None:
        """Convert a Mido message without importing Mido into the adapter core.

        Program/bank changes are intentionally ignored: the reference patch is
        selected explicitly, rather than changed by an arbitrary MIDI file.
        """
        if message.type not in {"note_on", "note_off", "control_change", "pitchwheel"}:
            return None
        return cls(
            sample, message.type, channel=message.channel,
            note=getattr(message, "note", 0), velocity=getattr(message, "velocity", 0),
            control=getattr(message, "control", 0), value=getattr(message, "value", 0),
            pitch=getattr(message, "pitch", 0),
        )
