"""
Core functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
import numpy as np

def exp_sigmoid(x, exponent = 10.0, max_value = 2.0, threshold = 1e-7):
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

def get_fft_size(frame_size, ir_size, power_of_two = True):
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

def crop_and_compensate_delay(audio, audio_size, ir_size, delay_compensation, padding = 'same'):
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

def overlap_and_add(frames, frame_step):
    """
    Reconstructs a signal from a framed representation, recreation of tf.signal.overlap_and_add
    
    Args:
        signal: A [batch_size, frames, frame_length] tensor of floats.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
        
    Returns:
        A 1D tensor of the reconstructed signal.
    """
    # Dimensions
    overlap_add_filter = torch.eye(frames.shape[-1], requires_grad = False).unsqueeze(1)
    output_signal = nn.functional.conv_transpose1d(frames.transpose(1, 2), 
                                                    overlap_add_filter, 
                                                    stride = frame_step, 
                                                    padding = 0).squeeze(1)

    return output_signal

def pad_axis(x, padding=(0, 0), axis=0, **pad_kwargs):
    """
    Pads only one axis of a tensor.

    Args:
        x: Input tensor.
        padding: Tuple of number of samples to pad (before, after).
        axis: Which axis to pad.
        **pad_kwargs: Other kwargs to pass to tf.pad.

    Returns:
        A tensor padded with padding along axis.
    """
    if axis >= len(x.shape):
        raise ValueError('Axis {} is out of bounds for tensor of dimension {}.'
                        .format(axis, len(x.shape)))
    n_end_dims = len(x.shape) - axis - 1
    paddings = (0,0) * n_end_dims + padding
    return F.pad(x, paddings, **pad_kwargs)

def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = torch.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator

def upsample_with_windows(inputs: torch.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> torch.Tensor:
    """Upsample a series of frames using using overlapping hann windows.

    Good for amplitude envelopes.
    Args:
        inputs: Framewise 3-D tensor. Shape [batch_size, n_channels, n_frames].
        n_timesteps: The time resolution of the output signal.
        add_endpoint: Hold the last timestep for an additional step as the endpoint.
        Then, n_timesteps is divided evenly into n_frames segments. If false, use
        the last timestep as the endpoint, producing (n_frames - 1) segments with
        each having a length of n_timesteps / (n_frames - 1).

    Returns:
        Upsampled 3-D tensor. Shape [batch_size, n_channels, n_timesteps].

    Raises:
        ValueError: If input does not have 3 dimensions.
        ValueError: If attempting to use function for downsampling.
        ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
        true) or n_frames - 1 (if add_endpoint is false).
    """
    inputs = torch.FloatTensor(inputs)

    if len(inputs.shape) != 3:
        raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
                        'not {}.'.format(inputs.shape))

    # Mimic behavior of tf.image.resize.
    # For forward (not endpointed), hold value for last interval.
    if add_endpoint:
        inputs = torch.cat((inputs, inputs[:, :, -1:]), dim=2)

    n_frames = int(inputs.shape[2])
    n_intervals = (n_frames - 1)

    if n_frames >= n_timesteps:
        raise ValueError('Upsample with windows cannot be used for downsampling'
                        'More input frames ({}) than output timesteps ({})'.format(
                            n_frames, n_timesteps))

    if n_timesteps % n_intervals != 0.0:
        minus_one = '' if add_endpoint else ' - 1'
        raise ValueError(
            'For upsampling, the target the number of timesteps must be divisible '
            'by the number of input frames{}. (timesteps:{}, frames:{}, '
            'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
                                    add_endpoint))

    # Constant overlap-add, half overlapping windows.
    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length)  # [window]

    # Broadcast multiply.
    # Add dimension for windows [batch_size, n_channels, n_frames, window].
    x = inputs[:, :, :, None]
    window = window[None, None, None, :]
    x_windowed = (x * window)
    # Collapse channel into batch size
    x_windowed = torch.reshape(x_windowed, (-1, x_windowed.shape[2], x_windowed.shape[3]))
    x = overlap_and_add(x_windowed, hop_size)
    # Reshape back to original shape.
    x = torch.reshape(x, (inputs.shape[0], inputs.shape[1], -1))

    # Trim the rise and fall of the first and last window.
    return x[:, :, hop_size:-hop_size]