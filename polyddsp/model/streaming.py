"""Controls-to-audio streaming core, independent of pitch and audio encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from polyddsp.model.additive import AdditiveSynth
from polyddsp.model.decoder import VoiceDecoder
from polyddsp.model.noise import FilteredNoise, NoiseState
from polyddsp.model.parity_ops import scale_f0_hz
from polyddsp.model.reverb import Reverb, ReverbState

if TYPE_CHECKING:
    from polyddsp.model.polyddsp import PolyDDSP


@dataclass(frozen=True)
class SynthesisState:
    hidden: torch.Tensor | None
    phase: torch.Tensor | None
    pitch: torch.Tensor
    harmonics: torch.Tensor
    amplitudes: torch.Tensor
    core_start: int
    harmonic_queue: torch.Tensor
    noise_queue: torch.Tensor
    noise: NoiseState | None
    reverb: ReverbState | None


class StreamingSynthesizer(nn.Module):
    """Drive trained synthesis components with allocated control frames.

    No pitch encoder, audio feature encoder, MIDI parser, or audio-device I/O is
    owned by this module. Callers supply pitch, velocity, normalized loudness,
    and optional z, on a common frame grid. Voice slot identities must persist
    between calls. Weights and scales must stay fixed while a state is in use.
    """

    def __init__(
        self,
        decoder: VoiceDecoder,
        additive: AdditiveSynth,
        noise_synth: FilteredNoise | None = None,
        reverb: Reverb | None = None,
        *,
        z_scale: float = 1.0,
        noise_scale: float = 1.0,
        noise_seed: int | None = None,
    ) -> None:
        super().__init__()
        if decoder.n_harmonics != additive.n_harmonics:
            raise ValueError("decoder and harmonic synthesizer dimensions must match")
        if noise_synth is not None and (
            noise_synth.frame_hop != additive.frame_hop or noise_synth.n_bands != decoder.n_bands
        ):
            raise ValueError("decoder and noise synthesizer timing/dimensions must match")
        self.decoder = decoder
        self.additive = additive
        self.noise_synth = noise_synth
        self.reverb = reverb
        self.z_scale = z_scale
        self.noise_scale = noise_scale
        self.noise_seed = noise_seed

    @classmethod
    def from_model(cls, model: PolyDDSP, *, noise_seed: int | None = None) -> StreamingSynthesizer:
        """Reuse a loaded model's trained synthesis modules and checkpoint scales."""
        return cls(
            model.decoder, model.additive,
            model.noise_synth if model.use_noise else None,
            model.reverb if model.use_reverb else None,
            z_scale=model.inference_z_scale, noise_scale=model.inference_noise_scale,
            noise_seed=noise_seed,
        )

    @property
    def sample_rate(self) -> int:
        return self.additive.sr

    @property
    def frame_hop(self) -> int:
        return self.additive.frame_hop

    def _mix(
        self, harmonic: torch.Tensor, noise: torch.Tensor,
        harmonic_queue: torch.Tensor, noise_queue: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        harmonic = torch.cat([harmonic_queue, harmonic], dim=-1)
        if self.noise_synth is None:
            return harmonic, harmonic[:, :0].clone(), noise[:, :0].clone()
        noise = torch.cat([noise_queue, noise], dim=-1)
        ready = min(harmonic.shape[-1], noise.shape[-1])
        return (
            harmonic[:, :ready] + self.noise_scale * noise[:, :ready],
            harmonic[:, ready:].clone(), noise[:, ready:].clone(),
        )

    @torch.no_grad()
    def process(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        loudness: torch.Tensor,
        z: torch.Tensor | None = None,
        state: SynthesisState | None = None,
        *,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SynthesisState]:
        """Return ready audio and next state for newly supplied control frames.

        pitch and velocity: (B, V, frames), pitch in Hz. loudness: (B, frames),
        scaled as (db/80)+1. z: (B, frames, z_dim) for z-enabled models.
        Optional explicit noise: (B, V, frames * frame_hop).

        None starts/resets a stream. Empty frame blocks are accepted. Harmonic
        interpolation holds one future control frame; noise delay compensation
        and reverb block buffering are aligned internally. Output lengths may
        vary: consume the returned audio in order, then call flush at EOF.
        """
        if pitch.ndim != 3:
            raise ValueError("pitch must have shape (batch, voices, frames)")
        B, V, count = pitch.shape
        if velocity.shape != pitch.shape or loudness.shape != (B, count):
            raise ValueError("velocity/loudness must match the pitch frame grid")
        if self.decoder.use_z and (z is None or z.shape != (B, count, self.decoder.z_dim)):
            raise ValueError("z-enabled models require z with shape (batch, frames, z_dim)")
        controls = [pitch, velocity, loudness] + ([z] if self.decoder.use_z else [])
        if any(t.device != pitch.device or t.dtype != pitch.dtype for t in controls):
            raise ValueError("all controls must have the same device and dtype")
        if noise is not None and (
            self.noise_synth is None or noise.shape != (B, V, count * self.additive.frame_hop)
        ):
            raise ValueError("explicit noise requires an enabled noise path and matching samples")

        empty = pitch.new_empty(B, 0)
        if state is None:
            state = SynthesisState(
                None, None, pitch[:, :, :0].clone(),
                pitch.new_empty(B, V, 0, self.decoder.n_harmonics),
                pitch[:, :, :0].clone(), 0, empty, empty, None, None,
            )
        elif (state.pitch.shape[:2] != (B, V) or state.pitch.device != pitch.device
              or state.pitch.dtype != pitch.dtype):
            raise ValueError("batch, voice slots, device, and dtype must remain fixed per stream")
        if count == 0:
            return empty, state

        harmonics, amplitudes, noise_mags, hidden = self.decoder.forward_with_state(
            scale_f0_hz(pitch), velocity, loudness, z, self.z_scale, state.hidden,
        )
        context_pitch = torch.cat([state.pitch, pitch], dim=2)
        context_harm = torch.cat([state.harmonics, harmonics], dim=2)
        context_amp = torch.cat([state.amplitudes, amplitudes], dim=2)
        end = context_pitch.shape[-1] - 1  # last frame is interpolation lookahead
        phase = state.phase
        if end > state.core_start:
            harmonic, phase = self.additive(
                context_pitch, context_harm, context_amp,
                initial_phase=phase, return_phase=True,
                keep_frames=(state.core_start, end),
            )
        else:
            harmonic = empty
        noise_state = state.noise
        noise_audio = empty
        if self.noise_synth is not None:
            excitation = None if noise is None else noise.reshape(B * V, -1)
            folded, noise_state = self.noise_synth.stream(
                noise_mags.reshape(B * V, count, self.decoder.n_bands), noise_state,
                noise=excitation, seed=self.noise_seed if noise_state is None else None,
            )
            noise_audio = folded.reshape(B, V, folded.shape[-1]).sum(dim=1)
        dry, harmonic_queue, noise_queue = self._mix(
            harmonic, noise_audio, state.harmonic_queue, state.noise_queue,
        )
        reverb_state = state.reverb
        output = dry
        if self.reverb is not None and dry.shape[-1]:
            output, reverb_state = self.reverb.stream(dry, reverb_state)
        keep = max(0, end - 1)
        next_state = SynthesisState(
            hidden.clone(), None if phase is None else phase.clone(),
            context_pitch[:, :, keep:].clone(), context_harm[:, :, keep:].clone(),
            context_amp[:, :, keep:].clone(), end - keep,
            harmonic_queue, noise_queue, noise_state, reverb_state,
        )
        return output, next_state

    @torch.no_grad()
    def flush(self, state: SynthesisState, *, include_tail: bool = False) -> torch.Tensor:
        """Finish the final control frame and optionally the full reverb tail.

        Default total output length is supplied_frames * frame_hop, matching
        clip synthesis. The final harmonic frame duplicates its endpoint just
        as offline synthesis does. No new controls or random noise are created.
        Discard state after flushing. include_tail affects reverb decay only.
        """
        B, V, frames = state.pitch.shape
        if frames == 0:
            return state.harmonic_queue.new_empty(B, 0)
        harmonic, _ = self.additive(
            state.pitch, state.harmonics, state.amplitudes,
            initial_phase=state.phase, return_phase=True,
            keep_frames=(state.core_start, frames),
        )
        noise = harmonic[:, :0]
        if self.noise_synth is not None and state.noise is not None:
            folded = self.noise_synth.flush(state.noise)
            noise = folded.reshape(B, V, folded.shape[-1]).sum(dim=1)
        dry, harmonic_queue, noise_queue = self._mix(
            harmonic, noise, state.harmonic_queue, state.noise_queue,
        )
        if harmonic_queue.shape[-1] or noise_queue.shape[-1]:
            raise RuntimeError("unmatched harmonic/noise samples at end of stream")
        if self.reverb is None:
            return dry
        output, reverb_state = self.reverb.stream(dry, state.reverb)
        return torch.cat([output, self.reverb.flush(reverb_state, include_tail=include_tail)], dim=-1)
