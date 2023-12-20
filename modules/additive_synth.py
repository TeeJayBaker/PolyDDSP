"""
Additive Harmonic Synthesiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
import numpy as np
import audio_ops as ops

class AdditiveSynth(nn.Module):
    """
    Additive Harmonic synthesiser

    Args:
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        device: Specify whether computed on cpu, cuda or mps

    Input: Synthesiser parameters coefficients of size (batch, frames, banks)
    Output: Filtered noise audio (batch, samples)
    """

    def __init__(self, sample_rate = 16000, 
                 normalize_below_nyquist=True,
                 amp_resample_method='window',
                 use_angular_cumsum=False, 
                 frame_length=64, 
                 attenuate_gain=0.02, 
                 device="mps"):
        
        super(AdditiveSynth, self).__init__()

        self.sample_rate = sample_rate
        self.normalize_below_nyquist = normalize_below_nyquist
        self.amp_resample_method = amp_resample_method
        self.use_angular_cumsum = use_angular_cumsum
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        self.device = device

    def angular_cumsum(self, angular_frequency, chunk_size=1000):
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
        chunks = torch.reshape(angular_frequency,
                            (n_batch, n_chunks, chunk_size) + (-1,) * n_ch_dims)
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
    
    def get_harmonic_frequencies(self, frequencies, n_harmonics):
        """Create integer multiples of the fundamental frequency.

        Args:
            frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
            n_harmonics: Number of harmonics.

        Returns:
            harmonic_frequencies: Oscillator frequencies (Hz).
            Shape [batch_size, :, n_harmonics].
        """
        frequencies = torch.FloatTensor(frequencies)

        f_ratios = torch.linspace(1.0, float(n_harmonics), int(n_harmonics))
        f_ratios = f_ratios[None, None, :]
        harmonic_frequencies = frequencies * f_ratios
        return harmonic_frequencies
          
    def remove_above_nyquist(self, frequency_envelopes,
                             amplitude_envelopes,
                             sample_rate = 16000):
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
        frequency_envelopes = torch.FloatTensor(frequency_envelopes)
        amplitude_envelopes = torch.FloatTensor(amplitude_envelopes)

        amplitude_envelopes = torch.where(
            torch.gt(frequency_envelopes, sample_rate / 2.0),
            torch.zeros_like(amplitude_envelopes), amplitude_envelopes)
        return amplitude_envelopes
    
    def normalize_harmonics(self, harmonic_distribution, f0_hz=None, sample_rate=None):
        """Normalize the harmonic distribution, optionally removing above nyquist."""
        # Bandlimit the harmonic distribution.
        if sample_rate is not None and f0_hz is not None:
            n_harmonics = int(harmonic_distribution.shape[-1])
            harmonic_frequencies = self.get_harmonic_frequencies(f0_hz, n_harmonics)
            harmonic_distribution = self.remove_above_nyquist(
                harmonic_frequencies, harmonic_distribution, sample_rate)

        # Normalize
        harmonic_distribution = ops.safe_divide(
            harmonic_distribution,
            torch.sum(harmonic_distribution, dim=-1, keepdim=True))
        return harmonic_distribution