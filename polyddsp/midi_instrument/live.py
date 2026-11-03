"""Live MIDI input with a muted FluidSynth reference and PolyDDSP-only playback."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import json
import math
from queue import Empty, Full, Queue
from time import perf_counter, sleep

import numpy as np

from polyddsp.midi_instrument._cli import add_model_arguments, make_instrument, optional_module
from polyddsp.midi_instrument.audio import AudioRingBuffer, LinearOutputResampler
from polyddsp.midi_instrument.events import MidiEvent


class MidiScheduler:
    """Bounded thread-safe input queue with stable ordering and late-event accounting."""

    def __init__(self, capacity: int = 4096) -> None:
        self.queue: Queue[MidiEvent] = Queue(maxsize=capacity)
        self._pending: list[MidiEvent] = []
        self.overflowed = False
        self.late_events = 0

    def put(self, event: MidiEvent) -> None:
        try:
            self.queue.put_nowait(event)
        except Full:
            self.overflowed = True

    def take(self, start: int, end: int) -> list[MidiEvent]:
        if self.overflowed:
            raise RuntimeError("MIDI event queue overflowed; stopping rather than dropping note-off events")
        while True:
            try:
                self._pending.append(self.queue.get_nowait())
            except Empty:
                break
        if len(self._pending) > self.queue.maxsize:
            raise RuntimeError("too many pending future MIDI events")
        self._pending.sort(key=lambda event: event.sample)
        ready = []
        split = 0
        for event in self._pending:
            if event.sample >= end:
                break
            if event.sample < start:
                self.late_events += 1
                event = replace(event, sample=start)
            ready.append(event)
            split += 1
        self._pending = self._pending[split:]
        return ready


def select_midi_port(mido, requested: str | None = None) -> str:
    """Choose the first input, excluding the system's MIDI Through loopback."""
    ports = mido.get_input_names()
    if requested is not None:
        if requested not in ports:
            raise ValueError(f"MIDI input {requested!r} is unavailable. Available inputs: {ports}")
        return requested
    candidates = [port for port in ports if "midi through" not in port.casefold()]
    if not candidates:
        raise RuntimeError(
            "No MIDI input device found. Connect a MIDI controller and retry, "
            "or use --list-midi-ports and --midi-port to select a virtual input."
        )
    return candidates[0]


def run_live(
    instrument, *, midi_port: str | None = None, audio_device=None,
    block_frames: int = 4, output_rate: int | None = None, buffer_blocks: int = 2,
    channel: int | None = None, seconds: float | None = None,
) -> dict:
    """Run inference on the main worker; the audio callback only copies ready samples.

    Extra buffering and sample-rate conversion are included in this prototype's
    playback path, so core benchmark timing alone is not its end-to-end latency.
    """
    if block_frames < 1 or buffer_blocks < 1 or (
        seconds is not None and (seconds <= 0 or not math.isfinite(seconds))
    ):
        raise ValueError("block_frames, buffer_blocks, and optional seconds must be finite and positive")
    if channel is not None and not 0 <= channel <= 15:
        raise ValueError("channel must be in [0, 15]")
    mido, sd = optional_module("mido"), optional_module("sounddevice")
    mido.set_backend("mido.backends.rtmidi")
    midi_port = select_midi_port(mido, midi_port)
    print(f"MIDI input: {midi_port}")
    info = sd.query_devices(audio_device, "output")
    rate = output_rate or round(info["default_samplerate"])
    resampler = LinearOutputResampler(instrument.sample_rate, rate)
    sd.check_output_settings(device=audio_device, channels=1, dtype="float32", samplerate=rate)
    size = block_frames * instrument.frame_hop
    output_size = math.ceil(size * rate / instrument.sample_rate)
    ring = AudioRingBuffer(buffer_blocks * output_size)
    scheduler = MidiScheduler()
    # Explicit startup silence primes playback while the first control blocks are rendered.
    ring.write(np.zeros(output_size, dtype=np.float32))
    origin = perf_counter()
    callback_errors: Queue[Exception] = Queue(maxsize=1)
    device_statuses = 0
    timings = deque(maxlen=4096)
    deadline_misses = 0
    maximum_ms = 0.0
    budget_ms = size * 1000 / instrument.sample_rate

    def midi_callback(message):
        try:
            if channel is not None and getattr(message, "channel", None) != channel:
                return
            sample = max(0, round((perf_counter() - origin) * instrument.sample_rate))
            event = MidiEvent.from_message(message, sample)
            if event is not None:
                scheduler.put(event)
        except Exception as exc:
            try:
                callback_errors.put_nowait(exc)
            except Full:
                pass

    def audio_callback(outdata, frames, time_info, status):
        nonlocal device_statuses
        if status:
            device_statuses += 1
        ring.read_into(outdata[:, 0])

    print(f"MIDI → PolyDDSP at {instrument.sample_rate} Hz; playback at {rate} Hz. Ctrl-C to stop.")
    try:
        with mido.open_input(midi_port, callback=midi_callback):
            with sd.OutputStream(
                samplerate=rate, blocksize=output_size, device=audio_device,
                channels=1, dtype="float32", latency="low", callback=audio_callback,
            ) as output_stream:
                while seconds is None or perf_counter() - origin < seconds:
                    if not callback_errors.empty():
                        raise RuntimeError("MIDI input callback failed") from callback_errors.get_nowait()
                    if not output_stream.active:
                        raise RuntimeError("audio output stream stopped unexpectedly")
                    if ring.free < output_size:
                        sleep(0.001)
                        continue
                    start = perf_counter()
                    events = scheduler.take(instrument.sample_position, instrument.sample_position + size)
                    audio = instrument.process(size, events)
                    converted = resampler.process(audio)
                    np.clip(converted, -1.0, 1.0, out=converted)
                    elapsed_ms = (perf_counter() - start) * 1000
                    timings.append(elapsed_ms)
                    maximum_ms = max(maximum_ms, elapsed_ms)
                    deadline_misses += elapsed_ms > budget_ms
                    written = 0
                    while written < len(converted):
                        written += ring.write(converted[written:])
                        if written < len(converted):
                            if not output_stream.active:
                                raise RuntimeError("audio output stopped while waiting for buffer space")
                            sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        instrument.release_all()
    return {
        "model_block_ms": budget_ms, "output_sample_rate": rate,
        "worker_median_ms_recent": float(np.median(timings)) if timings else 0.0,
        "worker_max_ms": maximum_ms, "deadline_misses": deadline_misses,
        "audio_queue_underruns": ring.underruns, "device_statuses": device_statuses,
        "late_midi_events": scheduler.late_events,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="streaming checkpoint (.pt), with config.yaml beside it")
    add_model_arguments(parser, required=False)
    parser.add_argument("--midi-port")
    parser.add_argument("--audio-device", help="device index or SoundDevice name query")
    parser.add_argument("--output-rate", type=int, help="defaults to the output device's sample rate")
    parser.add_argument("--buffer-blocks", type=int, default=2)
    parser.add_argument("--channel", type=int, help="optional zero-based source MIDI channel")
    parser.add_argument("--seconds", type=float, help="optional bounded playback duration")
    parser.add_argument("--list-midi-ports", action="store_true")
    parser.add_argument("--list-audio-devices", action="store_true")
    args = parser.parse_args(argv)
    if args.list_midi_ports:
        mido = optional_module("mido")
        mido.set_backend("mido.backends.rtmidi")
        print("\n".join(mido.get_input_names()))
        return
    if args.list_audio_devices:
        print(optional_module("sounddevice").query_devices())
        return
    if args.model and args.ckpt:
        parser.error("supply the model as a positional argument or --ckpt, not both")
    args.ckpt = args.model or args.ckpt
    if not args.ckpt or not args.soundfont:
        parser.error("MODEL and --soundfont are required for playback")
    device = int(args.audio_device) if args.audio_device and args.audio_device.isdigit() else args.audio_device
    try:
        with make_instrument(args) as instrument:
            stats = run_live(
                instrument, midi_port=args.midi_port, audio_device=device,
                block_frames=args.block_frames, output_rate=args.output_rate,
                buffer_blocks=args.buffer_blocks, channel=args.channel, seconds=args.seconds,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
