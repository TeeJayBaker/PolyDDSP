"""
Additive Harmonic Synthesiser
"""

import torch
import torch.nn as nn
import numpy as np
import modules.operations as ops
from typing import Optional


class AdditiveSynth(nn.Module):
    """
    Additive Harmonic synthesiser

    Args:
        sample_rate: sample rate of audio
        normalize_below_nyquist: whether to normalize above nyquist
        amp_resample_method: method of resampling amplitude envelopes,
            'window', 'linear', 'cubic' and 'nearest'
        use_angular_cumsum: whether to use angular cumulative sum
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        device: Specify whether computed on cpu, cuda or mps

    Input: Synthesiser parameters coefficients in a dictionary
        frequencies: Frequency features of size (batch, voices, frames)
        harmonics: Harmonics spectra (batch, voices, harmonics, frames)
        amplitude: per voice amplitude envelope (batch, voices, frames)
        noise: Noise filter coefficients of size (batch, filter_coeff, frames)
    Output: Filtered noise audio (batch, samples)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        normalize_below_nyquist: bool = True,
        amp_resample_method: str = "window",
        use_angular_cumsum: bool = False,
        frame_length: int = 64,
        attenuate_gain: float = 0.02,
        device: str = "mps",
    ):
        super(AdditiveSynth, self).__init__()

        self.sample_rate = sample_rate
        self.normalize_below_nyquist = normalize_below_nyquist
        self.amp_resample_method = amp_resample_method
        self.use_angular_cumsum = use_angular_cumsum
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        self.to(device)

    def angular_cumsum(
        self, angular_frequency: torch.Tensor, chunk_size: int = 1000
    ) -> torch.Tensor:
        """
        Get phase by cumulative sumation of angular frequency.

        Custom cumsum splits first axis into chunks to avoid accumulation error.
        Just taking tf.sin(tf.cumsum(angular_frequency)) leads to accumulation of
        phase errors that are audible for long segments or at high sample rates. Also,
        in reduced precision settings, cumsum can overflow the threshold.

        During generation, if syntheiszed examples are longer than ~100k samples,
        consider using angular_sum to avoid noticible phase errors. This version is
        currently activated by global gin injection. Set the gin parameter
        `oscillator_bank.use_angular_cumsum=True` to activate.

        Given that we are going to take the sin of the accumulated phase anyways, we
        don't care about the phase modulo 2 pi. This code chops the incoming frequency
        into chunks, applies cumsum to each chunk, takes mod 2pi, and then stitches
        them back together by adding the cumulative values of the final step of each
        chunk to the next chunk.

        Seems to be ~30% faster on CPU, but at least 40% slower on TPU.

        Args:
            angular_frequency: Radians per a sample. Shape [batch, time, ...].
                If there is no batch dimension, one will be temporarily added.
            chunk_size: Number of samples per a chunk. to avoid overflow at low
                precision [chunk_size <= (accumulation_threshold / pi)].

        Returns:
            The accumulated phase in range [0, 2*pi], shape [batch, time, ...].
        """

        # Get tensor shapes.
        n_batch = angular_frequency.shape[0]
        n_time = angular_frequency.shape[1]
        n_dims = len(angular_frequency.shape)
        n_ch_dims = n_dims - 2

        remainder = n_time % chunk_size
        if remainder:
            pad_amount = chunk_size - remainder
            angular_frequency = ops.pad_axis(angular_frequency, (0, pad_amount), axis=1)

        # Split input into chunks.
        length = angular_frequency.shape[1]
        n_chunks = int(length / chunk_size)
        chunks = torch.reshape(
            angular_frequency, (n_batch, n_chunks, chunk_size) + (-1,) * n_ch_dims
        )
        phase = torch.cumsum(chunks, dim=2)

        # Add offsets.
        # Offset of the next row is the last entry of the previous row.
        offsets = phase[:, :, -1:, ...] % (2.0 * np.pi)
        offsets = ops.pad_axis(offsets, (1, 0), axis=1)
        offsets = offsets[:, :-1, ...]

        # Offset is cumulative among the rows.
        offsets = torch.cumsum(offsets, dim=1) % (2.0 * np.pi)
        phase = phase + offsets

        # Put back in original shape.
        phase = phase % (2.0 * np.pi)
        phase = torch.reshape(phase, (n_batch, length) + (-1,) * n_ch_dims)

        # Remove padding if added it.
        if remainder:
            phase = phase[:, :n_time]

        return phase

    def get_harmonic_frequencies(
        self, frequencies: torch.Tensor, n_harmonics: int
    ) -> torch.Tensor:
        """Create integer multiples of the fundamental frequency.

        Args:
            frequencies: Fundamental frequencies (Hz). Shape [batch_size, n_voices, n_frames].
            n_harmonics: Number of harmonics.

        Returns:
            harmonic_frequencies: Oscillator frequencies (Hz).
            Shape [batch_size, n_voices, n_frames, n_harmonics].
        """
        # frequencies = torch.FloatTensor(frequencies)
        frequencies = frequencies.unsqueeze(-1)

        f_ratios = torch.linspace(
            1.0, float(n_harmonics), int(n_harmonics), device=frequencies.device
        )
        f_ratios = f_ratios[None, None, None, :]
        harmonic_frequencies = frequencies * f_ratios
        return harmonic_frequencies

    def remove_above_nyquist(
        self,
        frequency_envelopes: torch.Tensor,
        amplitude_envelopes: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """
        Set amplitudes for oscillators above nyquist to 0.

        Args:
            frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
                [batch_size, n_samples, n_sinusoids].
            amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
                n_samples, n_sinusoids].
            sample_rate: Sample rate in samples per a second.

        Returns:
            amplitude_envelopes: Sample-wise filtered oscillator amplitude.
                Shape [batch_size, n_samples, n_sinusoids].
        """
        # frequency_envelopes = torch.FloatTensor(frequency_envelopes)
        # amplitude_envelopes = torch.FloatTensor(amplitude_envelopes)

        amplitude_envelopes = torch.where(
            torch.gt(frequency_envelopes, sample_rate / 2.0),
            torch.zeros_like(amplitude_envelopes),
            amplitude_envelopes,
        )
        return amplitude_envelopes

    def normalize_harmonics(
        self,
        harmonic_distribution: torch.Tensor,
        f0_hz: torch.Tensor = None,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Normalize the harmonic distribution, optionally removing above nyquist."""
        # Bandlimit the harmonic distribution.
        if sample_rate is not None and f0_hz is not None:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = self.get_harmonic_frequencies(f0_hz, n_harmonics)
            harmonic_distribution = self.remove_above_nyquist(
                harmonic_frequencies, harmonic_distribution, sample_rate
            )

        # Normalize
        harmonic_distribution = ops.safe_divide(
            harmonic_distribution,
            torch.sum(harmonic_distribution, dim=-1, keepdim=True),
        )
        return harmonic_distribution

    def resample(
        self,
        inputs: torch.Tensor,
        n_timesteps: int,
        method: str = "linear",
        add_endpoint: bool = True,
    ) -> torch.Tensor:
        """Interpolates a tensor from n_frames to n_timesteps.

        Args:
            inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
            [batch_size, n_frames], [batch_size, channels, n_frames], or
            [batch_size, channels, n_frames, n_freq].
            n_timesteps: Time resolution of the output signal.
            method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
            'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
            'window' uses overlapping windows (only for upsampling) which is smoother
            for amplitude envelopes with large frame sizes.
            add_endpoint: Hold the last timestep for an additional step as the endpoint.
            Then, n_timesteps is divided evenly into n_frames segments. If false, use
            the last timestep as the endpoint, producing (n_frames - 1) segments with
            each having a length of n_timesteps / (n_frames - 1).
        Returns:
            Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
            [batch_size, n_timesteps], [batch_size, channels, n_timesteps], or
            [batch_size, channels, n_timesteps, n_freqs].

        Raises:
            ValueError: If method is 'window' and input is 4-D.
            ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
            'window'.
        """

        # inputs = torch.FloatTensor(inputs)
        is_1d = len(inputs.shape) == 1
        is_2d = len(inputs.shape) == 2
        is_4d = len(inputs.shape) == 4

        # Ensure inputs are at least 3d.
        if is_1d:
            inputs = inputs[None, None, :]
        elif is_2d:
            inputs = inputs[:, None, :]

        def resize(method):
            """Closure around torch.nn.Upsample."""
            # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
            outputs = inputs[:, :, :, None] if not is_4d else inputs
            outputs = nn.Upsample(
                size=[n_timesteps, outputs.shape[3]],
                mode=method,
                align_corners=not add_endpoint,
            )(outputs)
            return outputs[:, :, :, 0] if not is_4d else outputs

        if method == "nearest":
            outputs = resize("nearest")
        elif method == "linear":
            outputs = resize("bilinear")
        elif method == "cubic":
            outputs = resize("bicubic")
        elif method == "window":
            # squash to 3d if 4d
            if is_4d:
                batch, channels, _, n_freqs = inputs.shape
                inputs = inputs.permute(0, 3, 1, 2).reshape(
                    batch * n_freqs, channels, -1
                )

            outputs = ops.upsample_with_windows(inputs, n_timesteps, add_endpoint)

            # reconstruct 4d
            if is_4d:
                outputs = outputs.reshape(batch, n_freqs, channels, -1).permute(
                    0, 2, 3, 1
                )
        else:
            raise ValueError(
                "Method ({}) is invalid. Must be one of {}.".format(
                    method, "['nearest', 'linear', 'cubic', 'window']"
                )
            )

        # Return outputs to the same dimensionality of the inputs.
        if is_1d:
            outputs = outputs[0, 0, :]
        elif is_2d:
            outputs = outputs[:, 0, :]

        return outputs

    def oscillator_bank(
        self,
        frequency_envelopes: torch.Tensor,
        amplitude_envelopes: torch.Tensor,
        sample_rate: int = 16000,
        sum_sinusoids: bool = True,
        use_angular_cumsum: bool = False,
    ) -> torch.Tensor:
        """Generates audio from sample-wise frequencies for a bank of oscillators.

        Args:
            frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
            [batch_size, n_samples, n_sinusoids].
            amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
            n_samples, n_sinusoids].
            sample_rate: Sample rate in samples per a second.
            sum_sinusoids: Add up audio from all the sinusoids.
            use_angular_cumsum: If synthesized examples are longer than ~100k audio
            samples, consider use_angular_cumsum to avoid accumulating noticible phase
            errors due to the limited precision of tf.cumsum. Unlike the rest of the
            library, this property can be set with global dependency injection with
            gin. Set the gin parameter `oscillator_bank.use_angular_cumsum=True`
            to activate. Avoids accumulation of errors for generation, but don't use
            usually for training because it is slower on accelerators.

        Returns:
            wav: Sample-wise audio. Shape [batch_size, n_samples, n_sinusoids] if
            sum_sinusoids=False, else shape is [batch_size, n_samples].
        """
        # frequency_envelopes = torch.FloatTensor(frequency_envelopes)
        # amplitude_envelopes = torch.FloatTensor(amplitude_envelopes)

        # Don't exceed Nyquist.
        amplitude_envelopes = self.remove_above_nyquist(
            frequency_envelopes, amplitude_envelopes, sample_rate
        )

        # Angular frequency, Hz -> radians per sample.
        omegas = frequency_envelopes * (2.0 * np.pi)  # rad / sec
        omegas = omegas / float(sample_rate)  # rad / sample

        # Accumulate phase and synthesize.
        if use_angular_cumsum:
            # Avoids accumulation errors.
            phases = self.angular_cumsum(omegas)
        else:
            phases = torch.cumsum(omegas, dim=1)

        # Convert to waveforms.
        wavs = torch.sin(phases)
        audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
        if sum_sinusoids:
            audio = torch.sum(audio, dim=-1)  # [mb, n_samples]
        return audio

    def harmonic_synthesis(
        self,
        frequencies: torch.Tensor,
        amplitudes: torch.Tensor,
        harmonic_shifts: Optional[torch.Tensor] = None,
        harmonic_distribution: Optional[torch.Tensor] = None,
        n_samples: int = 64000,
        sample_rate: int = 16000,
        amp_resample_method: str = "window",
        use_angular_cumsum: bool = False,
    ) -> torch.Tensor:
        """Generate audio from frame-wise monophonic harmonic oscillator bank.

        Args:
            frequencies: Frame-wise fundamental frequency in Hz. Shape [batch_size, n_voices
            n_frames].
            amplitudes: Frame-wise oscillator peak amplitude. Shape [batch_size, n_voices
            n_frames].
            harmonic_shifts: Harmonic frequency variations (Hz), zero-centered. Total
            frequency of a harmonic is equal to (frequencies * harmonic_number * (1 +
            harmonic_shifts)). Shape [batch_size, n_voices, n_harmonics, n_frames].
            harmonic_distribution: Harmonic amplitude variations, ranged zero to one.
            Total amplitude of a harmonic is equal to (amplitudes *
            harmonic_distribution). Shape [batch_size, n_voices, n_harmonics, n_frames].
            n_samples: Total length of output audio. Interpolates and crops to this.
            sample_rate: Sample rate.
            amp_resample_method: Mode with which to resample amplitude envelopes.
            use_angular_cumsum: Use angular cumulative sum on accumulating phase
            instead of tf.cumsum. More accurate for inference.

        Returns:
            audio: Output audio. Shape [batch_size, n_samples]
        """
        # frequencies = torch.FloatTensor(frequencies)
        # amplitudes = torch.FloatTensor(amplitudes)

        if harmonic_distribution is not None:
            # harmonic_distribution = torch.FloatTensor(harmonic_distribution)
            harmonic_distribution = harmonic_distribution.permute(0, 1, 3, 2)
            n_harmonics = int(harmonic_distribution.shape[-1])
        elif harmonic_shifts is not None:
            # harmonic_shifts = torch.FloatTensor(harmonic_shifts)
            n_harmonics = int(harmonic_shifts.shape[-1])
        else:
            n_harmonics = 1

        # Create harmonic frequencies [batch_size, n_voices, n_frames, n_harmonics].
        harmonic_frequencies = self.get_harmonic_frequencies(frequencies, n_harmonics)
        if harmonic_shifts is not None:
            harmonic_frequencies *= 1.0 + harmonic_shifts

        # Create harmonic amplitudes [batch_size, n_voices, n_frames, n_harmonics].
        if harmonic_distribution is not None:
            harmonic_amplitudes = amplitudes.unsqueeze(-1) * harmonic_distribution
        else:
            harmonic_amplitudes = amplitudes.unsqueeze(-1)

        # Create sample-wise envelopes.
        frequency_envelopes = self.resample(
            harmonic_frequencies, n_samples
        )  # cycles/sec
        amplitude_envelopes = self.resample(
            harmonic_amplitudes, n_samples, method=amp_resample_method
        )

        # reshape voices into harmonics
        frequency_envelopes = frequency_envelopes.permute(0, 2, 1, 3).reshape(
            frequency_envelopes.shape[0], frequency_envelopes.shape[2], -1
        )
        amplitude_envelopes = amplitude_envelopes.permute(0, 2, 1, 3).reshape(
            amplitude_envelopes.shape[0], amplitude_envelopes.shape[2], -1
        )
        # Synthesize from harmonics [batch_size, n_samples].
        audio = self.oscillator_bank(
            frequency_envelopes,
            amplitude_envelopes,
            sample_rate=sample_rate,
            use_angular_cumsum=use_angular_cumsum,
        )
        return audio

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        n_samples = int(np.ceil(x["pitch"].shape[-1] * self.frame_length))

        audio = self.harmonic_synthesis(
            frequencies=x["pitch"],
            amplitudes=x["amplitude"],
            harmonic_shifts=None,
            harmonic_distribution=x["harmonics"],
            n_samples=n_samples,
            sample_rate=self.sample_rate,
            amp_resample_method=self.amp_resample_method,
            use_angular_cumsum=self.use_angular_cumsum,
        )
        return audio
