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

    def angular_cumsum(angular_frequency, chunk_size=1000):
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