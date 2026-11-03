"""Live instrument startup: explicit assets, checkpoint validation, automatic input."""
from types import SimpleNamespace

import pytest

from polyddsp.midi_instrument import live


@pytest.mark.parametrize("ports, requested, expected", [
    (["Keyboard"], None, "Keyboard"),
    (["Midi Through:Midi Through Port-0", "Keyboard"], None, "Keyboard"),
    (["Keyboard A", "Keyboard B"], None, "Keyboard A"),
    (["Keyboard A", "Keyboard B"], "Keyboard B", "Keyboard B"),
    (["Midi Through:Midi Through Port-0"], "Midi Through:Midi Through Port-0", "Midi Through:Midi Through Port-0"),
])
def test_auto_input_selection_and_explicit_override(ports, requested, expected):
    assert live.select_midi_port(SimpleNamespace(get_input_names=lambda: ports), requested) == expected


@pytest.mark.parametrize("ports", [[], ["Midi Through:Midi Through Port-0"]])
def test_missing_controller_has_actionable_error(ports):
    with pytest.raises(RuntimeError, match="Connect a MIDI controller"):
        live.select_midi_port(SimpleNamespace(get_input_names=lambda: ports))


def test_unavailable_requested_port_reports_available_inputs():
    with pytest.raises(ValueError, match="Keyboard A"):
        live.select_midi_port(SimpleNamespace(get_input_names=lambda: ["Keyboard A"]), "Keyboard B")


def test_positional_model_and_soundfont_start_playback(monkeypatch):
    calls = {}
    class Instrument:
        def __enter__(self): return self
        def __exit__(self, *exc): calls["closed"] = True
    def make(args):
        calls["ckpt"], calls["soundfont"] = args.ckpt, args.soundfont
        return Instrument()
    def run(instrument, **kwargs):
        calls["playback"] = kwargs
        return {}
    monkeypatch.setattr(live, "make_instrument", make)
    monkeypatch.setattr(live, "run_live", run)
    live.main(["run/best.pt", "--soundfont", "guitar.sf2"])
    assert calls["ckpt"] == "run/best.pt" and calls["soundfont"] == "guitar.sf2"
    assert calls["playback"]["midi_port"] is None
    assert calls["playback"]["audio_device"] is None
    assert calls["closed"]


@pytest.mark.parametrize("args", [
    ["run/best.pt"], ["--soundfont", "guitar.sf2"],
    ["run/best.pt", "--ckpt", "other.pt", "--soundfont", "guitar.sf2"],
])
def test_cli_requires_both_assets_and_one_model_argument(args, monkeypatch):
    monkeypatch.setattr(live, "make_instrument", lambda args: pytest.fail("must reject before loading"))
    with pytest.raises(SystemExit) as exc:
        live.main(args)
    assert exc.value.code == 2


def test_streaming_rejection_is_clean_and_precedes_playback(monkeypatch, capsys):
    from polyddsp.midi_instrument._cli import make_instrument
    def reject(*args, **kwargs):
        assert kwargs["require_streaming"] is True
        raise ValueError("Checkpoint is not a verified streaming model")
    monkeypatch.setattr("polyddsp.infer.load_run", reject)
    monkeypatch.setattr(live, "make_instrument", make_instrument)
    monkeypatch.setattr(live, "run_live", lambda *a, **kw: pytest.fail("must reject before playback"))
    with pytest.raises(SystemExit) as exc:
        live.main(["old.pt", "--soundfont", "guitar.sf2"])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "not a verified streaming model" in error and "Traceback" not in error
