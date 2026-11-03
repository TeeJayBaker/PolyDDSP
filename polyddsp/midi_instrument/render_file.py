"""Replay MIDI through a muted SoundFont and render only PolyDDSP to a WAV file."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from polyddsp.midi_instrument._cli import add_model_arguments, make_instrument, optional_module
from polyddsp.midi_instrument.engine import MidiInstrument
from polyddsp.midi_instrument.events import MidiEvent


def read_midi(path: str | Path, sample_rate: int, *, channel: int | None = None) -> tuple[list[MidiEvent], int]:
    """Convert tempo-aware MIDI-file timing to absolute model samples."""
    mido = optional_module("mido")
    if sample_rate <= 0 or (channel is not None and not 0 <= channel <= 15):
        raise ValueError("sample_rate must be positive and channel in [0, 15]")
    midi = mido.MidiFile(str(path))
    if midi.type == 2:
        raise ValueError("type-2 MIDI has independent track clocks; use a type-0 or type-1 file")
    elapsed = 0.0
    events = []
    for message in midi:
        elapsed += message.time
        if message.is_meta or (channel is not None and getattr(message, "channel", None) != channel):
            continue
        event = MidiEvent.from_message(message, round(elapsed * sample_rate))
        if event is not None:
            events.append(event)
    end = max(math.ceil(elapsed * sample_rate), events[-1].sample + 1 if events else 0)
    return events, end


def render_events(
    instrument: MidiInstrument, events: Iterable[MidiEvent], end_sample: int,
    *, block_frames: int = 4, release_seconds: float = 0.5, include_tail: bool = True,
) -> Iterator[np.ndarray]:
    """Use the same event/feature path as live input, without wall-clock sleeping."""
    if block_frames < 1 or release_seconds < 0 or not math.isfinite(release_seconds):
        raise ValueError("block_frames must be positive and release_seconds finite and nonnegative")
    events = sorted(events, key=lambda event: event.sample)
    if end_sample < instrument.sample_position or any(
        not instrument.sample_position <= event.sample < end_sample for event in events
    ):
        raise ValueError("all events must be in the requested playback interval")
    size = block_frames * instrument.frame_hop
    index = 0
    while instrument.sample_position < end_sample:
        end = min(end_sample, instrument.sample_position + size)
        start = index
        while index < len(events) and events[index].sample < end:
            index += 1
        yield instrument.process(end - instrument.sample_position, events[start:index])
    instrument.release_all()
    release_end = instrument.sample_position + round(release_seconds * instrument.sample_rate)
    while instrument.sample_position < release_end:
        yield instrument.process(min(size, release_end - instrument.sample_position))
    yield instrument.finish(include_tail=include_tail)


def render_file(
    instrument: MidiInstrument, midi_path: str | Path, output_path: str | Path,
    *, block_frames: int = 4, release_seconds: float = 0.5,
    include_tail: bool = True, channel: int | None = None,
) -> int:
    events, end = read_midi(midi_path, instrument.sample_rate, channel=channel)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("xb") as raw:
        with sf.SoundFile(raw, mode="w", samplerate=instrument.sample_rate, channels=1,
                          format="WAV", subtype="FLOAT") as writer:
            for audio in render_events(
                instrument, events, end, block_frames=block_frames,
                release_seconds=release_seconds, include_tail=include_tail,
            ):
                writer.write(audio)
                count += len(audio)
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--midi", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--channel", type=int, help="optional zero-based source MIDI channel")
    parser.add_argument("--release-seconds", type=float, default=0.5)
    parser.add_argument("--reverb-tail", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    with make_instrument(args) as instrument:
        count = render_file(
            instrument, args.midi, args.out, block_frames=args.block_frames,
            release_seconds=args.release_seconds, include_tail=args.reverb_tail, channel=args.channel,
        )
        print(f"Rendered {count / instrument.sample_rate:.3f}s at {instrument.sample_rate} Hz to {args.out}")


if __name__ == "__main__":
    main()
