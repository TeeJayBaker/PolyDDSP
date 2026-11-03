"""Instrument adapter tests without a proprietary/user-supplied SoundFont."""
from __future__ import annotations

from itertools import cycle
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from polyddsp.midi_instrument import FluidSynthRenderer, MidiEvent, MidiInstrument
from polyddsp.midi_instrument.render_file import read_midi, render_events, render_file
from polyddsp.midi_instrument.voices import VoiceAllocator
from polyddsp.model.polyddsp import PolyDDSP


class FakeRenderer:
    """Deterministic, timestamp-aware reference synth with the production protocol."""

    def __init__(self, voices=2):
        self.sample_rate, self.n_voices = 16000, voices
        self.position = 0
        self.notes = {}
        self.bends = {}
        self.commands = []
        self.audio = []
        self.closed = False

    def note_on(self, slot, note, velocity):
        self.notes[slot] = (note, velocity)
        self.commands.append((self.position, "on", slot, note))

    def note_off(self, slot, note):
        self.notes.pop(slot, None)
        self.commands.append((self.position, "off", slot, note))

    def stop(self, slot):
        self.notes.pop(slot, None)
        self.commands.append((self.position, "stop", slot))

    def pitch_bend(self, slot, value):
        self.bends[slot] = value

    def control_change(self, slot, control, value):
        self.commands.append((self.position, "cc", slot, control, value))

    def render(self, samples):
        times = np.arange(self.position, self.position + samples) / self.sample_rate
        output = np.zeros(samples, dtype=np.float64)
        for slot, (note, velocity) in self.notes.items():
            bend = self.bends.get(slot, 0) * 2 / 8192
            hz = 440 * 2 ** ((note + bend - 69) / 12)
            output += np.sin(2 * np.pi * hz * times) * velocity / 127 * 0.1
        self.position += samples
        output = output.astype(np.float32)
        self.audio.append(output)
        return output

    def close(self):
        self.closed = True


def tiny_model(use_z=True, noise=True, reverb=True):
    torch.manual_seed(83)
    return PolyDDSP(
        n_voices=2, use_z=use_z, use_noise=noise, use_reverb=reverb,
        n_harmonics=4, z_dim=4, reverb_len=129,
        gru_hidden=16, mlp_hidden=16, mlp_layers=1, pitch_source="cached_basic_pitch",
    ).eval()


def performance_events():
    return [
        MidiEvent(0, "note_on", note=60, velocity=100),
        MidiEvent(63, "note_on", note=64, velocity=80),
        MidiEvent(64, "pitchwheel", pitch=4096),
        MidiEvent(128, "control_change", control=64, value=127),
        MidiEvent(129, "note_off", note=60),
        MidiEvent(200, "note_on", note=67, velocity=64),
        MidiEvent(257, "control_change", control=64, value=0),
        MidiEvent(300, "note_off", note=67),
    ]


def run_chunks(model, sizes):
    renderer = FakeRenderer()
    instrument = MidiInstrument(model, renderer, noise_seed=89)
    events = performance_events()
    parts = []
    for size in cycle(sizes):
        start = instrument.sample_position
        if start == 1027:
            break
        end = min(start + size, 1027)
        block_events = [event for event in events if start <= event.sample < end]
        parts.append(instrument.process(end - start, block_events))
    parts.append(instrument.finish())
    return np.concatenate(parts), renderer, instrument


@pytest.mark.parametrize("use_z,noise,reverb", [(True, True, True), (True, False, False), (False, True, True)])
@pytest.mark.parametrize("sizes", [(1,), (13, 51, 0, 197), (256,)])
def test_adapter_output_and_events_are_independent_of_input_chunking(use_z, noise, reverb, sizes):
    model = tiny_model(use_z, noise, reverb)
    expected, reference_renderer, _ = run_chunks(model, (1027,))
    actual, renderer, instrument = run_chunks(model, sizes)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    assert renderer.commands == reference_renderer.commands
    assert len(actual) == 1088 + (128 if reverb else 0)
    assert instrument.sample_position == renderer.position == 1088
    assert len(instrument._loudness_state.audio_buffer[0]) < 512


def test_reference_features_match_offline_and_only_model_audio_is_returned(monkeypatch):
    model = tiny_model(noise=False, reverb=False)
    def no_transcription(*args, **kwargs):
        raise AssertionError("MIDI must not call AMT")
    monkeypatch.setattr(model.pitch_encoder, "forward", no_transcription)
    renderer = FakeRenderer()
    instrument = MidiInstrument(model, renderer)
    controls = []
    original = instrument.synth.process
    def capture(pitch, velocity, loudness, z, state):
        controls.append((pitch.clone(), velocity.clone(), loudness.clone(), z.clone()))
        return original(pitch, velocity, loudness, z, state)
    monkeypatch.setattr(instrument.synth, "process", capture)
    parts = [instrument.process(1280, performance_events()), instrument.finish(include_tail=False)]
    proxy = torch.from_numpy(np.concatenate(renderer.audio)).unsqueeze(0)
    pitch, velocity, loudness, z = controls[0]
    with torch.no_grad():
        torch.testing.assert_close(loudness, model.loudness(proxy))
        torch.testing.assert_close(z, model.z_encoder(proxy))
        expected, _ = model.synthesize(pitch, velocity, loudness, z)
    actual = np.concatenate(parts)
    np.testing.assert_allclose(actual, expected[0].numpy(), rtol=2e-4, atol=2e-5)
    assert np.max(np.abs(actual - proxy[0].numpy())) > 1e-3


def test_event_on_hop_boundary_belongs_to_next_control_frame(monkeypatch):
    instrument = MidiInstrument(tiny_model(noise=False, reverb=False), FakeRenderer())
    captured = []
    original = instrument.synth.process
    def capture(pitch, *args):
        captured.append(pitch.clone())
        return original(pitch, *args)
    monkeypatch.setattr(instrument.synth, "process", capture)
    instrument.process(128, [MidiEvent(64, "note_on", note=69, velocity=127)])
    instrument.process(64, [MidiEvent(128, "note_off", note=69)])
    assert captured[0][0, 0].tolist() == [0.0, 440.0]
    assert captured[1][0, 0, 0] == 0
    assert (64, "on", 0, 69) in instrument.renderer.commands
    assert (128, "off", 0, 69) in instrument.renderer.commands


def test_voice_sustain_pitch_bend_and_quietest_voice_stealing():
    renderer = FakeRenderer()
    voices = VoiceAllocator(renderer)
    for event in performance_events()[:7]:
        voices.handle(event)
    pitch, velocity = voices.snapshot()
    assert pitch[0] == 0  # pedal release freed the sustained C
    assert pitch[1] == pytest.approx(440 * 2 ** ((68 - 69) / 12))  # G plus one-semitone bend
    assert velocity[1] == pytest.approx(64 / 127)
    assert [v.note if v else None for v in voices.voices] == [None, 67]
    assert sum(command[1] == "stop" for command in renderer.commands) == 3
    voices.all_notes_off()
    assert not voices.snapshot()[0].any()
    assert not renderer.notes


def test_same_note_on_different_channels_and_retrigger():
    voices = VoiceAllocator(FakeRenderer())
    voices.handle(MidiEvent(0, "note_on", channel=0, note=60, velocity=100))
    voices.handle(MidiEvent(0, "note_on", channel=1, note=60, velocity=80))
    voices.handle(MidiEvent(1, "note_on", channel=0, note=60, velocity=90))
    assert len([v for v in voices.voices if v]) == 2
    voices.handle(MidiEvent(2, "note_on", channel=0, note=60, velocity=0))
    assert voices.voices[0] is None
    assert voices.voices[1].channel == 1
    voices.handle(MidiEvent(3, "control_change", channel=1, control=120))
    assert not voices.snapshot()[0].any()


def test_adapter_rejects_out_of_interval_events_and_ends_cleanly():
    renderer = FakeRenderer()
    with MidiInstrument(tiny_model(), renderer) as instrument:
        assert instrument.process(0).size == 0
        with pytest.raises(ValueError, match="interval"):
            instrument.process(64, [MidiEvent(64, "note_on", note=60, velocity=100)])
        assert instrument.sample_position == 0
        instrument.process(1, [MidiEvent(0, "note_on", note=60, velocity=100)])
        instrument.release_all()
        assert instrument.finish(include_tail=False).shape == (64,)
        with pytest.raises(RuntimeError, match="finished"):
            instrument.process(64)
    assert renderer.closed


class NativeSynth:
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.selected = []
        self.ccs = []
        self.deleted = False

    def start(self):
        raise AssertionError("reference synth must never start an audio device")

    def sfload(self, path): return 1
    def program_select(self, channel, sfid, bank, preset):
        self.selected.append((channel, sfid, bank, preset))
        return 0
    def cc(self, channel, controller, value): self.ccs.append((channel, controller, value))
    def noteon(self, channel, note, velocity): pass
    def noteoff(self, channel, note): pass
    def pitch_bend(self, channel, value): pass
    def get_samples(self, samples): return np.tile(np.array([32767, -32768], dtype=np.int16), samples)
    def delete(self): self.deleted = True


def test_fluidsynth_wrapper_is_headless_and_normalizes_stereo(tmp_path, monkeypatch):
    import polyddsp.midi_instrument.renderer as module

    native = NativeSynth()
    monkeypatch.setattr(module, "import_module", lambda name: SimpleNamespace(Synth=lambda **kwargs: native))
    patch = tmp_path / "guitar.sf2"
    patch.touch()
    renderer = FluidSynthRenderer(patch, 16000, 10, preset=24)
    assert [selection[0] for selection in native.selected] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    assert (0, 6, 2) in native.ccs  # bend range explicitly set to two semitones
    renderer.note_on(0, 60, 127)
    np.testing.assert_array_equal(renderer.render(3), np.full(3, -1 / 65536, dtype=np.float32))
    assert renderer.render(0).size == 0
    renderer.close()
    assert native.deleted
    with pytest.raises(RuntimeError, match="closed"):
        renderer.render(1)


def test_renderer_reports_missing_native_library_and_failed_patch(tmp_path, monkeypatch):
    import polyddsp.midi_instrument.renderer as module

    patch = tmp_path / "guitar.sf2"
    patch.touch()
    def unavailable(name): raise ImportError("no native library")
    monkeypatch.setattr(module, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="native FluidSynth"):
        FluidSynthRenderer(patch, 16000, 2)
    native = NativeSynth()
    native.sfload = lambda path: -1
    monkeypatch.setattr(module, "import_module", lambda name: SimpleNamespace(Synth=lambda **kwargs: native))
    with pytest.raises(RuntimeError, match="could not load"):
        FluidSynthRenderer(patch, 16000, 2)
    assert native.deleted


def test_midi_file_tempo_changes_and_channel_filter(tmp_path):
    mido = pytest.importorskip("mido")
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.Message("program_change", program=99),
        mido.Message("note_on", note=60, velocity=100),
        mido.MetaMessage("set_tempo", tempo=1000000, time=480),
        mido.Message("note_off", note=60, time=480),
        mido.MetaMessage("end_of_track", time=240),
    ])
    path = tmp_path / "tempo.mid"
    midi.save(path)
    events, end = read_midi(path, 16000)
    assert [(event.kind, event.sample) for event in events] == [("note_on", 0), ("note_off", 24000)]
    assert end == 32000
    assert read_midi(path, 16000, channel=1) == ([], 32000)


def test_midi_file_render_writes_model_only_float_wav(tmp_path):
    mido = pytest.importorskip("mido")
    midi = mido.MidiFile()
    midi.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=60, velocity=100),
        mido.Message("note_off", note=60, time=6),
    ]))
    source, target = tmp_path / "input.mid", tmp_path / "output.wav"
    midi.save(source)
    with MidiInstrument(tiny_model(noise=False, reverb=False), FakeRenderer()) as instrument:
        count = render_file(instrument, source, target, release_seconds=0.004)
    audio, rate = sf.read(target)
    assert count == len(audio) == 192
    assert rate == 16000
    assert sf.info(target).subtype == "FLOAT"
    assert np.isfinite(audio).all()
    with MidiInstrument(tiny_model(), FakeRenderer()) as instrument:
        with pytest.raises(FileExistsError):
            render_file(instrument, source, target)


def test_file_replay_includes_release_interval():
    with MidiInstrument(tiny_model(noise=False, reverb=False), FakeRenderer()) as instrument:
        audio = np.concatenate(list(render_events(
            instrument, [MidiEvent(0, "note_on", note=69, velocity=100)], 128,
            block_frames=1, release_seconds=0.004, include_tail=False,
        )))
        assert audio.shape == (192,)
        assert not instrument.allocator.snapshot()[0].any()
