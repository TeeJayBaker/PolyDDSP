"""Top-level PolyDDSP model."""
from __future__ import annotations


import torch
import torch.nn as nn
from omegaconf import DictConfig

from polyddsp.model.additive import AdditiveSynth
from polyddsp.model.decoder import VoiceDecoder
from polyddsp.model.loudness import LoudnessExtractor
from polyddsp.model.noise import FilteredNoise
from polyddsp.model.pitch import PitchEncoder
from polyddsp.model.reverb import Reverb
from polyddsp.model.z import ZEncoder


class PolyDDSP(nn.Module):
    """End-to-end polyphonic DDSP model.

    For V=1 (solo) the chain reduces to DDSP's solo_instrument recipe: one
    decoder emits joint (amp, harm_dist, noise_mags); harmonic + filtered-noise
    synths run; mix passes through a single learnable reverb IR.

    For V>1 the same decoder weights run on every voice slot via batch fold,
    so each voice gets its own (harm + noise) stem; stems sum into mix before
    reverb. The shared decoder forces gradient coupling between heads — both
    within a voice (amp/harm/noise share hidden state) and across voices (same
    weights) — which prevents the IR-as-amplifier degeneracy that an earlier
    design with one independent decoder per output head exhibited.
    """

    def __init__(
        self,
        sr: int = 16_000,
        frame_hop: int = 64,
        n_harmonics: int = 100,
        z_dim: int = 16,
        noise_bands: int = 65,
        noise_window: int = 0,
        reverb_len: int = 48_000,
        loudness_n_fft: int = 512,
        gru_hidden: int = 512,
        mlp_hidden: int = 512,
        mlp_layers: int = 3,
        n_voices: int = 1,
        use_z: bool = False,
        use_reverb: bool = True,
        use_noise: bool = True,
        pitch_source: str = "basic_pitch",
    ) -> None:
        super().__init__()
        self.sr = sr
        self.frame_hop = frame_hop
        self.n_voices = n_voices
        self.use_z = use_z
        self.use_reverb = use_reverb
        self.use_noise = use_noise

        self.pitch_encoder = PitchEncoder(
            sr=sr,
            n_voices=n_voices,
            target_frame_hop=frame_hop,
            source=pitch_source,
        )
        self.loudness = LoudnessExtractor(
            sr=sr,
            frame_hop=frame_hop,
            n_fft=loudness_n_fft,
        )
        self.z_encoder = ZEncoder(sr=sr, frame_hop=frame_hop, z_dim=z_dim) if use_z else None
        # Solo (n_voices=1): velocity is constant 1.0, drop the input stream
        # to match DDSP's `RnnFcDecoder.input_keys = ('ld_scaled', 'f0_scaled')`.
        # Poly: per-voice velocity from BP `y_n[freq_idx]` is load-bearing.
        use_velocity = n_voices > 1
        self.decoder = VoiceDecoder(
            z_dim=z_dim,
            n_harmonics=n_harmonics,
            n_bands=noise_bands,
            gru_hidden=gru_hidden,
            mlp_hidden=mlp_hidden,
            mlp_layers=mlp_layers,
            use_z=use_z,
            use_velocity=use_velocity,
        )
        self.additive = AdditiveSynth(
            sr=sr,
            frame_hop=frame_hop,
            n_harmonics=n_harmonics,
        )
        # Stateless — one instance reused across voices via batch fold.
        self.noise_synth = FilteredNoise(
            frame_hop=frame_hop, n_bands=noise_bands, window_size=noise_window
        )
        self.reverb = Reverb(reverb_len=reverb_len) if use_reverb else None

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "PolyDDSP":
        exp = cfg.experiment.model
        n_harmonics = exp.get("n_harmonics", cfg.model.n_harmonics)
        return cls(
            sr=cfg.model.sr,
            frame_hop=cfg.model.frame_hop,
            n_harmonics=n_harmonics,
            z_dim=cfg.model.z_dim,
            noise_bands=cfg.model.noise_bands,
            noise_window=cfg.model.noise_window,
            reverb_len=cfg.model.reverb_len,
            loudness_n_fft=cfg.model.get("loudness_n_fft", 512),
            gru_hidden=cfg.model.gru_hidden,
            mlp_hidden=cfg.model.mlp_hidden,
            mlp_layers=cfg.model.mlp_layers,
            n_voices=exp.n_voices,
            use_z=exp.use_z,
            use_reverb=exp.use_reverb,
            use_noise=exp.get("use_noise", True),
            pitch_source=exp.get("pitch_source", "basic_pitch"),
        )

    def _synthesise(
        self,
        pitch: torch.Tensor,
        velocity: torch.Tensor,
        loudness: torch.Tensor,
        z: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        """Decoder and synthesisers for a block of frames, before reverb.

        Args:
            pitch:    (B, V, T_frames) fundamental frequency in Hz (0 = silence)
            velocity: (B, V, T_frames) in [0, 1]
            loudness: (B, T_frames) normalised A-weighted loudness
            z:        (B, T_frames, z_dim) or None when `use_z=False`

        Returns `(mix, parts)` where `mix` is the dry harmonic+noise sum
        `(B, T_frames * frame_hop)` and `parts` holds the intermediates that
        `forward`'s aux dict reports.

        This is the whole-file path used by `forward` (training / evaluation on
        clips). `render` does not call it: rendering a whole recording has to
        block the oscillator bank, which means threading decoder and oscillator
        state across blocks — see `_decode_block` and `render`.
        """
        from polyddsp.model.parity_ops import scale_f0_hz

        # Decoder always sees a perceptually-linear pitch axis (hz_to_midi/127).
        # Synth always sees raw f0 in Hz so harmonic frequencies are correct.
        pitch_for_decoder = scale_f0_hz(pitch)

        harm_dist, amp_v, noise_mags = self.decoder(pitch_for_decoder, velocity, loudness, z)

        # Per-voice harmonic synth (already vectorised across V).
        audio_harm = self.additive(pitch=pitch, harm_dist=harm_dist, amp_v=amp_v)

        # Per-voice noise: batch-fold V into batch dim, run one stateless synth,
        # sum the V independent realisations in time — matches the data-generating
        # process for synthetic poly mixes (V independent solo recordings summed)
        # and reduces to standard DDSP noise for V=1.
        if self.use_noise:
            B, V, T_frames, n_bands = noise_mags.shape
            noise_v = self.noise_synth(noise_mags.reshape(B * V, T_frames, n_bands))
            T_samples = noise_v.shape[-1]
            audio_noise = noise_v.reshape(B, V, T_samples).sum(dim=1)
            audio_noise = audio_noise[..., : audio_harm.shape[-1]]
        else:
            audio_noise = torch.zeros_like(audio_harm)

        mix = audio_harm + audio_noise
        parts = dict(
            noise_mags=noise_mags,
            harm_dist=harm_dist,
            amp_v=amp_v,
            audio_harm=audio_harm,
            audio_noise=audio_noise,
        )
        return mix, parts

    def _decode_block(
        self,
        pitch_scaled: torch.Tensor,
        velocity: torch.Tensor,
        loudness: torch.Tensor,
        z: torch.Tensor | None,
        h_0: torch.Tensor | None,
        n_core: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`VoiceDecoder.forward` with the GRU's hidden state threaded through.

        `VoiceDecoder.forward` drops the GRU's final hidden state (`gru_out, _ =
        self.gru(cat)`), so calling it once per block would silently restart the
        recurrence at every block boundary — the decoder outputs would then
        differ from a whole-file pass no matter how exact the oscillator is.
        Rather than change `decoder.py` (whose `forward` signature is what
        training and the parity tests are written against), this method runs the
        decoder's own submodules in exactly the order `VoiceDecoder.forward` does
        and threads `h_0` / `h_n` through `self.decoder.gru` itself. Everything
        except the GRU is frame-wise, so nothing else needs state.
        `tests/test_infer.py::test_decode_block_matches_voice_decoder` pins the
        two paths together.

        `n_core` frames are the block proper; any frames beyond that are the
        one-frame look-ahead `AdditiveSynth`'s amplitude upsampler needs. The
        returned `h_n` is the state *after* frame `n_core - 1`, so the next
        block resumes on a contiguous frame grid — the look-ahead frame is
        computed from `h_n` but does not advance it. A GRU is strictly causal, so
        its output at the look-ahead frame is the whole-file output there.

        Returns `(harm_dist, amp_v, noise_mags, h_n)` over *all* supplied frames.
        """
        from polyddsp.model.decoder import _modified_sigmoid, _stack_voice_features

        dec = self.decoder
        B, V, T = pitch_scaled.shape
        vel_in = velocity if dec.use_velocity else None
        feat_in = _stack_voice_features(pitch_scaled, vel_in, loudness, z if dec.use_z else None)

        idx = 0
        feats = [dec.f0_mlp(feat_in[..., idx:idx + 1])]
        idx += 1
        if dec.use_velocity:
            feats.append(dec.vel_mlp(feat_in[..., idx:idx + 1]))
            idx += 1
        feats.append(dec.loud_mlp(feat_in[..., idx:idx + 1]))
        idx += 1
        if dec.use_z:
            feats.append(dec.z_mlp(feat_in[..., idx:]))
        cat = torch.cat(feats, dim=-1)

        gru_core, h_n = dec.gru(cat[:, :n_core], h_0)
        if T > n_core:
            gru_ahead, _ = dec.gru(cat[:, n_core:], h_n)
            gru_out = torch.cat([gru_core, gru_ahead], dim=1)
        else:
            gru_out = gru_core
        post = dec.post_mlp(torch.cat([cat, gru_out], dim=-1))
        head_out = dec.head(post)  # (B*V, T, 1+H+n_bands)

        amp_v = _modified_sigmoid(head_out[..., 0]).reshape(B, V, T)
        harm_dist = head_out[..., 1:1 + dec.n_harmonics].reshape(B, V, T, dec.n_harmonics)
        noise_mags = head_out[..., 1 + dec.n_harmonics:].reshape(B, V, T, dec.n_bands)
        return harm_dist, amp_v, noise_mags, h_n

    def _conditioning(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor | None,
        velocity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
        """Pitch / loudness / z over the whole input, trimmed to a common frame count."""
        pitch_out = self.pitch_encoder(
            audio, pitch_hint=pitch, velocity_hint=velocity,
        )
        loudness = self.loudness(audio, normalise=True)
        z = self.z_encoder(audio) if self.use_z else None

        target_frames = audio.shape[-1] // self.frame_hop
        pitch = pitch_out["pitch"][..., :target_frames]
        velocity = pitch_out["velocity"][..., :target_frames]
        loudness = loudness[..., :target_frames]
        if z is not None:
            z = z[:, :target_frames, :]
        return pitch, velocity, loudness, z, pitch_out["bp_post"]

    def forward(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        z_scale: float | None = 1.0
    ) -> tuple[torch.Tensor, dict]:
        pitch, velocity, loudness, z, bp_post = self._conditioning(audio, pitch, velocity)

        mix, parts = self._synthesise(pitch, velocity, loudness, z, z_scale)

        out = self.reverb(mix) if self.use_reverb else mix
        out = out[..., : audio.shape[-1]]

        aux = dict(
            bp_post=bp_post,
            pitch=pitch,
            velocity=velocity,
            loudness=loudness,
            z=z,
            **parts,
        )
        return out, aux

    @torch.no_grad()
    def render(
        self,
        audio: torch.Tensor,
        pitch: torch.Tensor | None = None,
        velocity: torch.Tensor | None = None,
        chunk_frames: int = 1000,
    ) -> torch.Tensor:
        """Resynthesise arbitrary-length audio without a whole-file synth pass.

        `forward` is fine for training clips but cannot render a whole recording:
        `AdditiveSynth` holds several `(B, V*n_harmonics, T_samples)` fp32
        tensors, ~16 kB of working set per audio sample at V=10 / H=100, so a
        few minutes of audio needs tens of GB. `render` splits the chain by cost
        and blocks only the expensive stage, carrying exact state across blocks
        so the result matches a whole-file `forward` numerically rather than
        approximately:

        * **Conditioning** (pitch encoder, loudness, z) runs once over the whole
          input, exactly as `forward` does, so note timing and the loudness
          envelope stay globally consistent instead of restarting per block.
          This is still O(T) memory — a very long input needs a couple of GB —
          just with a far smaller constant than the oscillator bank.
        * **Decoder + harmonic synth** run on consecutive blocks of
          `chunk_frames` frames. The decoder GRU's hidden state is threaded from
          block to block (`_decode_block`) and the oscillator phase is carried as
          `initial_phase` / `final_phase` (`AdditiveSynth`, mod 2π so fp32 does
          not lose resolution over a long file). Each block is handed one frame
          of look-behind and one of look-ahead — `AdditiveSynth.keep_frames`
          explains why the two upsamplers need them — and returns exactly its
          own `chunk_frames * frame_hop` samples, identical to a whole-file
          pass.
        * **Filtered noise** runs *once*, whole-file, after the loop: only the
          small `(B, V, T_frames, n_bands)` magnitudes are accumulated per block.
          `FilteredNoise` is `(B*V, T_frames, frame_hop)`-shaped internally
          (~192 MB at V=10 for five minutes, so it fits), and running it once
          avoids per-frame overlap-add seams at the block joins entirely.
        * **Reverb** is applied once to the fully assembled dry mix. `Reverb` is
          LTI, so one FFT convolution over the whole dry signal is exactly a
          whole-file pass, whereas per-block reverb would truncate each block's
          3-second IR tail at the block boundary.

        `FilteredNoise` draws fresh uniform noise on every call, so with
        `use_noise=True` a render equals `forward` only statistically. With
        `use_noise=False` — and, for `use_noise=True`, against another `render`
        under the same RNG seed — the match is numerical: block-vs-whole-file
        differences are ~1e-6 peak, from fp32 rounding in the phase carry and in
        splitting the GRU scan, not from any seam.

        Args:
            audio: (B, T) input at `self.sr`.
            pitch, velocity: (B, V, T_frames) conditioning, required when
                `pitch_source="cached_basic_pitch"`.
            chunk_frames: frames of audio synthesised per block. The default
                1000 frames is 4 s at 16 kHz / hop 64 — the training horizon.
        Returns:
            (B, T) — the same length as `audio`. Because synthesis works on whole
            frames, a trailing partial frame (`T % frame_hop`) is emitted as
            zeros; feed frame-aligned audio if that matters.
        """
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be >= 1, got {chunk_frames}")

        from polyddsp.model.parity_ops import scale_f0_hz

        pitch, velocity, loudness, z, _ = self._conditioning(audio, pitch, velocity)
        pitch_scaled = scale_f0_hz(pitch)
        total_frames = pitch.shape[-1]
        hop = self.frame_hop
        B = audio.shape[0]
        T_out = total_frames * hop

        dry = audio.new_zeros(B, T_out)
        noise_mag_blocks: list[torch.Tensor] = []
        h: torch.Tensor | None = None
        phase: torch.Tensor | None = None
        tail: tuple[torch.Tensor, torch.Tensor] | None = None  # decoder out at frame `start-1`

        for start in range(0, total_frames, chunk_frames):
            end = min(start + chunk_frames, total_frames)
            n_core = end - start
            ahead = min(end + 1, total_frames)  # one look-ahead frame for the amp lattice

            z_block = z[:, start:ahead, :] if z is not None else None
            harm_dist, amp_v, noise_mags, h = self._decode_block(
                pitch_scaled[..., start:ahead],
                velocity[..., start:ahead],
                loudness[..., start:ahead],
                z_block,
                h,
                n_core,
            )
            noise_mag_blocks.append(noise_mags[:, :, :n_core, :])
            # Stash frame `end-1` before prepending: it is the next block's look-behind.
            next_tail = (harm_dist[:, :, n_core - 1:n_core, :], amp_v[:, :, n_core - 1:n_core])

            if tail is not None:
                harm_dist = torch.cat([tail[0], harm_dist], dim=2)
                amp_v = torch.cat([tail[1], amp_v], dim=2)
                lo = start - 1
            else:
                lo = start
            i0 = start - lo
            block, phase = self.additive(
                pitch=pitch[..., lo:ahead],
                harm_dist=harm_dist,
                amp_v=amp_v,
                initial_phase=phase,
                return_phase=True,
                keep_frames=(i0, i0 + n_core),
            )
            dry[:, start * hop : end * hop] = block
            tail = next_tail

        if self.use_noise and noise_mag_blocks:
            noise_mags = torch.cat(noise_mag_blocks, dim=2)
            _, V, T_frames, n_bands = noise_mags.shape
            noise_v = self.noise_synth(noise_mags.reshape(B * V, T_frames, n_bands))
            audio_noise = noise_v.reshape(B, V, -1).sum(dim=1)
            dry = dry + audio_noise[..., :T_out]

        out = self.reverb(dry) if self.use_reverb else dry
        T_in = audio.shape[-1]
        if out.shape[-1] < T_in:
            out = torch.cat([out, out.new_zeros(B, T_in - out.shape[-1])], dim=-1)
        return out[..., :T_in]
