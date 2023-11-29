"""
Core functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
import numpy as np

def exp_sigmoid(self, x, exponent = 10.0, max_value = 2.0, threshold = 1e-7):
    """
    Exponentiated Sigmoid pointwise nonlinearity.

    Bounds input to [threshold, max_value] with slope given by exponent.

    Args:
        x: Input tensor.
        exponent: In nonlinear regime (away from x=0), the output varies by this
        factor for every change of x by 1.0.
        max_value: Limiting value at x=inf.
        threshold: Limiting value at x=-inf. Stablizes training when outputs are
        pushed to 0.

    Returns:
        A tensor with pointwise nonlinearity applied.
    """
    x = x.float()
    return max_value * torch.sigmoid(x) ** torch.log(exponent) + threshold

def get_fft_size(self, frame_size, ir_size, power_of_two = True):
    """
    Get FFT size for given frame and IR sizes

    Args:
        frame_size: Size of the audio frame.
        ir_size: Size of the convolving impulse response.
        power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
        numbers. TPU requires power of 2, while GPU is more flexible.

    Returns:
        fft_size: Size for efficient FFT.
    """
    conv_frame_size = frame_size + ir_size - 1
    if power_of_two:
        fft_size = int(2**np.ciel(np.log2(conv_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(conv_frame_size))
    return fft_size

def crop_and_compensate_delay(self, audio, audio_size, ir_size, delay_compensation, padding = 'same'):
    """
    Crop audio to compensate for delay

    Args:
        audio: Audio after convolution. Tensor of shape [batch, time_steps].
        audio_size: Initial size of the audio before convolution.
        ir_size: Size of the convolving impulse response.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
        same size as the input audio (audio_timesteps). For 'valid' the audio is
        extended to include the tail of the impulse response (audio_timesteps +
        ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
        for group delay of the impulse response. If delay_compensation < 0 it
        defaults to automatically calculating a constant group delay of the
        windowed linear phase filter from frequency_impulse_response().

    Returns:
        Tensor of cropped and shifted audio.

    Raises:
        ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError('Padding must be \'valid\' or \'same\', instead '
                        'of {}.'.format(padding))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 -
            1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return audio[:, start:-end]

def overlap_and_add(self, frames, frame_step):
    """
    Reconstructs a signal from a framed representation, recreation of tf.signal.overlap_and_add
    
    Args:
        signal: A [batch_size, frames, frame_length] tensor of floats.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
        
    Returns:
        A 1D tensor of the reconstructed signal.
    """
    print(frames.shape)
    # Dimensions
    overlap_add_filter = torch.eye(frames.shape[-1], requires_grad = False).unsqueeze(1)
    output_signal = nn.functional.conv_transpose1d(frames.transpose(1, 2), 
                                                    overlap_add_filter, 
                                                    stride = frame_step, 
                                                    padding = 0).squeeze(1)

    return output_signal