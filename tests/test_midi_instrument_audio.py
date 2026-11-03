"""Playback utilities can be validated without opening an audio or MIDI device."""
from __future__ import annotations

from itertools import cycle
from types import SimpleNamespace

import numpy as np
import pytest

from polyddsp.midi_instrument.audio import AudioRingBuffer, LinearOutputResampler
from polyddsp.midi_instrument.events import MidiEvent
from polyddsp.midi_instrument.live import MidiScheduler, run_live


@pytest.mark.parametrize("output_rate", [16000, 44100, 48000])
@pytest.mark.parametrize("sizes", [(1,), (17, 0, 7), (999,)])
def test_output_resampling_is_chunk_independent_and_does_not_drift(output_rate, sizes):
    audio = np.random.default_rng(97).normal(size=129).astype(np.float32)
    resampler = LinearOutputResampler(16000, output_rate)
    parts = []
    consumed = 0
    for size in cycle(sizes):
        if consumed == len(audio):
            break
        chunk = audio[consumed:consumed + size]
        parts.append(resampler.process(chunk))
        consumed += len(chunk)
        assert len(resampler._history) <= 1
    parts.append(resampler.process(np.empty(0), final=True))
    actual = np.concatenate(parts)
    length = (len(audio) * output_rate + 15999) // 16000
    expected = np.interp(np.arange(length) * 16000 / output_rate, np.arange(len(audio)), audio)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(RuntimeError, match="finished"):
        resampler.process(audio)


def test_audio_ring_wraparound_partial_writes_and_underrun():
    ring = AudioRingBuffer(5)
    assert ring.write(np.arange(4, dtype=np.float32)) == 4
    output = np.empty(3, dtype=np.float32)
    ring.read_into(output)
    np.testing.assert_array_equal(output, [0, 1, 2])
    assert ring.write(np.arange(4, 10, dtype=np.float32)) == 4
    output = np.empty(5, dtype=np.float32)
    ring.read_into(output)
    np.testing.assert_array_equal(output, [3, 4, 5, 6, 7])
    ring.read_into(output)
    np.testing.assert_array_equal(output, np.zeros(5))
    assert ring.underruns == 1
    assert ring.free == 5


def test_audio_callback_does_not_wait_for_writer_lock():
    ring = AudioRingBuffer(5)
    output = np.ones(5, dtype=np.float32)
    with ring._lock:
        ring.read_into(output)
    assert not output.any()
    assert ring.underruns == 1


def test_midi_scheduler_clamps_late_events_and_retains_future_events():
    scheduler = MidiScheduler()
    scheduler.put(MidiEvent(20, "note_off", note=60))
    scheduler.put(MidiEvent(1, "note_on", note=60, velocity=100))
    ready = scheduler.take(5, 10)
    assert len(ready) == 1 and ready[0].sample == 5
    assert scheduler.late_events == 1
    assert not scheduler.take(10, 20)
    assert scheduler.take(20, 30)[0].kind == "note_off"


def test_midi_scheduler_reports_overflow_instead_of_dropping_releases():
    scheduler = MidiScheduler(capacity=1)
    scheduler.put(MidiEvent(0, "note_on", note=60, velocity=100))
    scheduler.put(MidiEvent(1, "note_off", note=60))
    with pytest.raises(RuntimeError, match="overflowed"):
        scheduler.take(0, 64)


def test_live_runtime_uses_worker_and_cleans_up_mocked_devices(monkeypatch):
    import polyddsp.midi_instrument.live as module

    clock = [0.0]
    streams = []
    played = []
    def now():
        clock[0] += 0.001
        return clock[0]
    class Output:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active = False
            streams.append(self)
        def __enter__(self): self.active = True; return self
        def __exit__(self, *exc): self.active = False
        def play(self):
            output = np.empty((self.kwargs["blocksize"], 1), dtype=np.float32)
            self.kwargs["callback"](output, len(output), None, False)
            played.append(output)
    class Port:
        def __init__(self, callback): self.callback = callback
        def __enter__(self):
            self.callback(SimpleNamespace(type="note_on", channel=0, note=69, velocity=100))
            return self
        def __exit__(self, *exc): pass
    class Instrument:
        sample_rate, frame_hop, sample_position = 16000, 64, 0
        released = False
        def process(self, size, events):
            self.sample_position += size
            return np.full(size, 0.25, dtype=np.float32)
        def release_all(self): self.released = True
    sd = SimpleNamespace(
        query_devices=lambda *args: {"default_samplerate": 48000},
        check_output_settings=lambda **kwargs: None, OutputStream=Output,
    )
    def open_input(port, callback):
        assert port == "Keyboard 0"
        return Port(callback)
    mido = SimpleNamespace(
        set_backend=lambda *args: None, open_input=open_input,
        get_input_names=lambda: ["Midi Through:Midi Through Port-0", "Keyboard 0"],
    )
    monkeypatch.setattr(module, "optional_module", lambda name: sd if name == "sounddevice" else mido)
    monkeypatch.setattr(module, "perf_counter", now)
    monkeypatch.setattr(module, "sleep", lambda seconds: streams[0].play())
    instrument = Instrument()
    stats = run_live(instrument, seconds=0.025)
    assert instrument.sample_position > 0
    assert stats["output_sample_rate"] == 48000
    assert instrument.released and not streams[0].active
    assert played and all(np.isfinite(audio).all() for audio in played)
