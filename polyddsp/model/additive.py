"""Additive harmonic synthesiser with exact per-sample phase accumulation."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi


def angular_cumsum(
    omegas: torch.Tensor,
    chunk_size: int = 1000,
    initial_phase: torch.Tensor | None = None,
    return_final: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Cumulative sum of angular frequency along dim=1, modding 2π between chunks.

    `omegas` is `(B, T, ...)`; the returned phase has the same shape and lies in
    `[0, 2π)`. This is the accumulator `AdditiveSynth` uses — two properties make
    it the right primitive rather than a plain `torch.cumsum`:

    * **Bounded magnitude.** fp32 resolution is relative, so a plain cumsum over
      a long file loses phase precision as the absolute phase grows: at 6 s /
      16 kHz the accumulated rounding on a 1.2 kHz partial is already ~2e-3 rad,
      and it keeps growing. Holding the running phase in `[0, 2π)` keeps the
      per-sample resolution constant no matter how long the file is.
    * **Split invariance.** A chunk's internal cumsum depends only on that
      chunk's samples, and the running inter-chunk offset is accumulated in
      fp64, so the result is (to ~1e-6 rad) independent of where the signal was
      cut: `angular_cumsum(x)` equals the concatenation of `angular_cumsum` over
      consecutive slices of `x`, each fed the previous slice's `final_phase` as
      its `initial_phase`, provided the cuts land on chunk boundaries. That is
      what lets `PolyDDSP.render` synthesise a long file block by block and get
      numerically the same samples as a whole-file `forward`. `AdditiveSynth`
      passes `chunk_size=frame_hop`, so every frame boundary — and therefore
      every possible block boundary — is chunk-aligned.

    Args:
        omegas: (B, T, ...) per-sample angular frequency increment in radians.
        chunk_size: samples per mod-2π chunk. Must divide any intended split
            point for split invariance to hold.
        initial_phase: (B, 1, ...) phase to start from — the preceding block's
            `final_phase`. `None` means start at 0.
        return_final: if True return `(phase, final_phase)`, where
            `final_phase` is `(B, 1, ...)`: the phase at the last sample taken
            mod 2π, ready to be passed as the next block's `initial_phase`.
            It is accumulated in fp64 internally so the carry itself does not
            become the dominant error term over many blocks.
    """
    n_batch, n_time = omegas.shape[0], omegas.shape[1]
    trailing = omegas.shape[2:]

    pad = (-n_time) % chunk_size
    if pad:
        omegas = F.pad(omegas.movedim(1, -1), (0, pad)).movedim(-1, 1)

    length = omegas.shape[1]
    n_chunks = length // chunk_size

    # `unflatten` is a view whenever dim 1 is internally contiguous — true for
    # the `(B, C, T) -> (B, T, C)` transposed view `AdditiveSynth` hands in — so
    # no copy of the (potentially multi-GB) omega tensor is made here, and the
    # scan below runs along the innermost (stride-1) axis.
    chunks = omegas.unflatten(1, (n_chunks, chunk_size))
    phase = torch.cumsum(chunks, dim=2)  # (B, n_chunks, chunk_size, ...)

    # Inter-chunk offsets: the *only* part of the sum whose length depends on how
    # the signal was split, so the only part that has to be split-invariant.
    # fp64 keeps it so; it is `chunk_size`× smaller than `phase`, so the cost is
    # a few percent of the working set.
    ends = phase[:, :, -1].double() % TWO_PI  # (B, n_chunks, ...)
    incl = torch.cumsum(ends, dim=1)
    excl = torch.cat([torch.zeros_like(incl[:, :1]), incl[:, :-1]], dim=1)
    if initial_phase is not None:
        excl = excl + initial_phase.double()
    excl = excl % TWO_PI

    # In-place so peak memory matches a plain `cumsum` + `sin`. Both ops have
    # gradient 1 w.r.t. `phase` and neither saves its output for backward, so
    # this stays autograd-correct.
    phase = phase.add_(excl.unsqueeze(2).to(phase.dtype)).remainder_(TWO_PI)
    phase = phase.reshape(n_batch, length, *trailing)
    if pad:
        phase = phase[:, :n_time]

    if not return_final:
        return phase

    final = incl[:, -1:]  # (B, 1, ...) — total phase advance, fp64
    if initial_phase is not None:
        final = final + initial_phase.double()
    final = (final % TWO_PI).to(omegas.dtype)
    return phase, final


def _upsample_linear(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Linear upsample along the last dimension.

    Matches DDSP `core.resample` (`tf.compat.v1.image.resize` with
    `align_corners=False`).
    """
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)


class AdditiveSynth(nn.Module):
    """Harmonic oscillator bank — DDSP-compatible.

    Inputs:
        pitch:     (B, V, T_frames) — fundamental frequency in Hz
        harm_dist: (B, V, T_frames, H) — raw logits; we apply
                   exp_sigmoid + normalize_harmonics + sum-to-one inside.
        amp_v:    (B, V, T_frames) — per-voice amplitude envelope

    Output: audio (B, T_samples) where T_samples = T_frames * frame_hop

    Single per-voice amplitude path matches DDSP's `ddsp.synths.Harmonic`.
    `MonoDecoder` emits one `amp_v` per voice; `harm_dist` and `noise_mags` come
    from the same shared head — see `decoder.py`.

    ## Block rendering

    A whole-file pass materialises several `(B, V*n_harmonics, T_samples)` fp32
    tensors, so long files must be synthesised in blocks (see `PolyDDSP.render`).
    `initial_phase` / `return_phase` carry the oscillator state across blocks and
    `keep_frames` handles the two upsamplers' lattice sensitivity, so a
    block-by-block render reproduces the whole-file samples rather than
    approximating them. See `keep_frames` for the frame-padding contract.
    """

    def __init__(
        self,
        sr: int = 16_000,
        frame_hop: int = 64,
        n_harmonics: int = 100,
    ) -> None:
        super().__init__()
        self.sr = sr
        self.frame_hop = frame_hop
        self.n_harmonics = n_harmonics

    def forward(
        self,
        pitch: torch.Tensor,
        harm_dist: torch.Tensor,
        amp_v: torch.Tensor,
        initial_phase: torch.Tensor | None = None,
        return_phase: bool = False,
        keep_frames: tuple[int, int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Synthesise `(B, n_kept_frames * frame_hop)` samples.

        Args:
            pitch, harm_dist, amp_v: control signals — see the class docstring.
            initial_phase: (B, 1, V*n_harmonics) phase added before `sin`, i.e.
                the phase the oscillators had at the sample immediately before
                this block. `None` starts from 0.
            return_phase: also return `(B, 1, V*n_harmonics)` — the phase at the
                last rendered sample, mod 2π. Feed it back as the next block's
                `initial_phase`.
            keep_frames: `(i0, i1)` — render only the samples belonging to
                supplied frames `[i0, i1)`, i.e. `(i1 - i0) * frame_hop`
                samples. `None` (the default) means all of them, which
                reproduces the plain whole-file behaviour.

                The frames outside `[i0, i1)` are *context*, not output. They
                are needed because both upsamplers are local but
                lattice-sensitive:

                * `upsample_with_windows` (amplitudes) overlap-adds Hann windows
                  of `2*frame_hop` at hop `frame_hop`, so output sample `t` in
                  `[k*hop, (k+1)*hop)` depends on frames `k` and `k+1`. Rendering
                  frames `[i0, i1)` therefore needs frame `i1` as well — one
                  frame of **look-ahead**. When `i1 == T_frames` there is none, so
                  we fall back to `add_endpoint=True`, exactly reproducing the
                  whole-file pass's duplication of the last frame.
                * `_upsample_linear` maps output sample `i` to input coordinate
                  `(i + 0.5)/hop - 0.5`, so the first sample of the block reaches
                  back to frame `i0 - 1`, and `F.interpolate` clamps at the ends
                  of whatever tensor it is handed. One frame of **look-behind**
                  makes every kept sample interior, so clamping cannot alter it.

                So `PolyDDSP.render` hands in frames `[s-1, e+1)` (clipped to the
                file) with `keep_frames` selecting `[s, e)` within them.
        """
        from polyddsp.model.noise import exp_sigmoid
        from polyddsp.model.parity_ops import normalize_harmonics, upsample_with_windows

        B, V, T_frames = pitch.shape
        assert harm_dist.shape == (B, V, T_frames, self.n_harmonics)
        assert amp_v.shape == (B, V, T_frames)
        i0, i1 = (0, T_frames) if keep_frames is None else keep_frames
        if not 0 <= i0 < i1 <= T_frames:
            raise ValueError(
                f"keep_frames={keep_frames} out of range for {T_frames} supplied frames"
            )
        hop = self.frame_hop
        n_chan = V * self.n_harmonics
        T_samples = (i1 - i0) * hop

        # exp_sigmoid + per-frame Nyquist mask + sum-to-one (DDSP normalize_harmonics).
        hd = exp_sigmoid(harm_dist)
        f0_b = pitch.unsqueeze(-1).reshape(B * V, T_frames, 1)
        hd_b = hd.reshape(B * V, T_frames, self.n_harmonics)
        hd_b = normalize_harmonics(hd_b, f0_b, sample_rate=self.sr)
        harm_dist = hd_b.reshape(B, V, T_frames, self.n_harmonics)

        amp_per_h = amp_v.unsqueeze(-1) * harm_dist  # (B, V, T_frames, H)

        ratios = torch.arange(1, self.n_harmonics + 1, device=pitch.device, dtype=pitch.dtype)
        harm_freqs = pitch.unsqueeze(-1) * ratios  # (B, V, T_frames, H)
        nyquist = self.sr / 2.0

        # align_corners=False matches DDSP's `tf.compat.v1.image.resize` lattice.
        # One frame of look-behind/look-ahead (when available) keeps every kept
        # sample interior, so `F.interpolate`'s end clamping can't perturb it.
        lo = max(0, i0 - 1)
        hi = min(T_frames, i1 + 1)
        hf = harm_freqs[:, :, lo:hi, :].permute(0, 1, 3, 2).reshape(B, n_chan, hi - lo)
        hf_s = _upsample_linear(hf, (hi - lo) * hop)
        skip = (i0 - lo) * hop
        harm_freqs_s = hf_s[..., skip : skip + T_samples]

        # Per-harmonic amp upsample uses overlap-add Hann windows (DDSP
        # `upsample_with_windows`). Sample `t` needs frames `t//hop` and
        # `t//hop + 1`, so use the look-ahead frame when there is one and fall
        # back to the whole-file `add_endpoint=True` duplication when there isn't.
        has_lookahead = i1 < T_frames
        amp_sl = amp_per_h[:, :, i0 : i1 + 1, :] if has_lookahead else amp_per_h[:, :, i0:i1, :]
        amp_btc = amp_sl.permute(0, 1, 3, 2).reshape(B, n_chan, amp_sl.shape[2])
        amp_per_h_s = upsample_with_windows(
            amp_btc.transpose(1, 2), T_samples, add_endpoint=not has_lookahead
        ).transpose(1, 2)

        # Sample-rate Nyquist mask (`>=`) — frame-rate mask is in normalize_harmonics,
        # but interpolation across frame boundaries can leak energy across Nyquist.
        amp_per_h_s = torch.where(
            harm_freqs_s >= nyquist,
            torch.zeros_like(amp_per_h_s),
            amp_per_h_s,
        )

        omegas = harm_freqs_s * (TWO_PI / self.sr)
        omegas = omegas.transpose(1, 2)  # (B, T_samples, V*H)
        # Frame-aligned chunks: any block boundary is a multiple of `frame_hop`,
        # so this accumulator is split-invariant at every possible block edge.
        out = angular_cumsum(
            omegas,
            chunk_size=hop,
            initial_phase=initial_phase,
            return_final=return_phase,
        )
        phase, final_phase = out if return_phase else (out, None)
        sin_phase = torch.sin(phase)

        sinusoids = sin_phase * amp_per_h_s.transpose(1, 2)
        audio = sinusoids.sum(dim=-1)
        if return_phase:
            return audio, final_phase
        return audio
