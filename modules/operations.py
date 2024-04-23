"""
Core functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
import numpy as np


def exp_sigmoid(
    x: torch.Tensor,
    exponent: float = 10.0,
    max_value: float = 2.0,
    threshold: float = 1e-7,
) -> torch.Tensor:
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
    exponent = torch.tensor(exponent, dtype=x.dtype)
    return max_value * torch.sigmoid(x) ** torch.log(exponent) + threshold


def get_fft_size(frame_size: int, ir_size: int, power_of_two: bool = True) -> int:
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
        fft_size = int(2 ** np.ceil(np.log2(conv_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(conv_frame_size))
    return fft_size


def crop_and_compensate_delay(
    audio: torch.Tensor,
    audio_size: int,
    ir_size: int,
    delay_compensation: int,
    padding: str = "same",
) -> torch.Tensor:
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
    if padding == "valid":
        crop_size = ir_size + audio_size - 1
    elif padding == "same":
        crop_size = audio_size
    else:
        raise ValueError(
            "Padding must be 'valid' or 'same', instead " "of {}.".format(padding)
        )

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = (ir_size - 1) // 2 - 1 if delay_compensation < 0 else delay_compensation
    end = crop - start
    return audio[:, start:-end]


def overlap_and_add(frames: torch.Tensor, frame_step: int) -> torch.Tensor:
    """
    Reconstructs a signal from a framed representation, recreation of tf.signal.overlap_and_add

    Args:
        signal: A [batch_size, frames, frame_length] tensor of floats.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A 1D tensor of the reconstructed signal.
    """
    # Dimensions
    overlap_add_filter = torch.eye(
        frames.shape[-1], requires_grad=False, device=frames.device
    ).unsqueeze(1)
    output_signal = nn.functional.conv_transpose1d(
        frames.transpose(1, 2), overlap_add_filter, stride=frame_step, padding=0
    ).squeeze(1)

    return output_signal


def pad_axis(
    x: torch.Tensor, padding: tuple[int] = (0, 0), axis: int = 0, **pad_kwargs
) -> torch.Tensor:
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
        raise ValueError(
            "Axis {} is out of bounds for tensor of dimension {}.".format(
                axis, len(x.shape)
            )
        )
    n_end_dims = len(x.shape) - axis - 1
    paddings = (0, 0) * n_end_dims + padding
    return F.pad(x, paddings, **pad_kwargs)


def safe_divide(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Avoid dividing by zero by adding a small epsilon."""
    safe_denominator = torch.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator


def fft_convolve(
    audio: torch.Tensor,
    impulse_response: torch.Tensor,
    padding: str = "same",
    delay_compensation: int = -1,
) -> torch.Tensor:
    """
    Filter audio with frames of time-varying impulse responses.

    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
        audio: Input audio. Tensor of shape [batch, audio_timesteps].
        impulse_response: Finite impulse response to convolve. Can either be a 2-D
            Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
            ir_frames, ir_size]. A 2-D tensor will apply a single linear
            time-invariant filter to the audio. A 3-D Tensor will apply a linear
            time-varying filter. Automatically chops the audio into equally shaped
            blocks to match ir_frames.
        padding: Either 'valid' or 'same'. For 'same' the final output to be the
            same size as the input audio (audio_timesteps). For 'valid' the audio is
            extended to include the tail of the impulse response (audio_timesteps +
            ir_timesteps - 1).
        delay_compensation: Samples to crop from start of output audio to compensate
            for group delay of the impulse response. If delay_compensation is less
            than 0 it defaults to automatically calculating a constant group delay of
            the windowed linear phase filter from frequency_impulse_response().

    Returns:
        audio_out: Convolved audio. Tensor of shape
            [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
            [batch, audio_timesteps] ('same' padding).

    Raises:
        ValueError: If audio and impulse response have different batch size.
        ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
            number of impulse response frames is on the order of the audio size and
            not a multiple of the audio size.)
    """
    # audio = torch.FloatTensor(audio)
    # impulse_response = torch.FloatTensor(impulse_response)

    # Get shapes of audio.
    batch_size, audio_size = list(audio.size())

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = list(impulse_response.size())
    if len(ir_shape) == 2:
        impulse_response = impulse_response.unsqueeze(1)

    # Broadcast impulse response.
    if ir_shape[0] == 1 and batch_size > 1:
        impulse_response = torch.tile(impulse_response, (batch_size, 1, 1))

    # Get shapes of impulse response.
    ir_shape = list(impulse_response.shape)
    batch_size_ir, n_ir_frames, ir_size = ir_shape

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError(
            "Batch size of audio ({}) and impulse response ({}) must "
            "be the same.".format(batch_size, batch_size_ir)
        )

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size

    # Pad audio to match frame size (and match tf.signal.frame())
    pad_size = (
        frame_size - abs(audio_size % hop_size) if audio_size % frame_size != 0 else 0
    )
    audio = F.pad(audio, (0, pad_size))
    audio_frames = audio.unfold(-1, frame_size, hop_size)

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            "Number of Audio frames ({}) and impulse response frames ({}) do not "
            "match. For small hop size = ceil(audio_size / n_ir_frames), "
            "number of impulse response frames must be a multiple of the audio "
            "size.".format(n_audio_frames, n_ir_frames)
        )

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_two=True)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(impulse_response, fft_size)

    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.mul(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    if audio_frames_out.shape[1] > 1:
        audio_out = overlap_and_add(audio_frames_out, hop_size)
    else:
        audio_out = audio_frames_out.squeeze(1)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(
        audio_out, audio_size, ir_size, delay_compensation, padding
    )


def upsample_with_windows(
    inputs: torch.Tensor, n_timesteps: int, add_endpoint: bool = True
) -> torch.Tensor:
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
    # inputs = torch.FloatTensor(inputs)

    if len(inputs.shape) != 3:
        raise ValueError(
            "Upsample_with_windows() only supports 3 dimensions, " "not {}.".format(
                inputs.shape
            )
        )

    # Mimic behavior of tf.image.resize.
    # For forward (not endpointed), hold value for last interval.
    if add_endpoint:
        inputs = torch.cat((inputs, inputs[:, :, -1:]), dim=2)

    n_frames = int(inputs.shape[2])
    n_intervals = n_frames - 1

    if n_frames >= n_timesteps:
        raise ValueError(
            "Upsample with windows cannot be used for downsampling"
            "More input frames ({}) than output timesteps ({})".format(
                n_frames, n_timesteps
            )
        )

    if n_timesteps % n_intervals != 0.0:
        minus_one = "" if add_endpoint else " - 1"
        raise ValueError(
            "For upsampling, the target the number of timesteps must be divisible "
            "by the number of input frames{}. (timesteps:{}, frames:{}, "
            "add_endpoint={}).".format(minus_one, n_timesteps, n_frames, add_endpoint)
        )

    # Constant overlap-add, half overlapping windows.
    hop_size = n_timesteps // n_intervals
    window_length = 2 * hop_size
    window = torch.hann_window(window_length, device=inputs.device)  # [window]

    # Broadcast multiply.
    # Add dimension for windows [batch_size, n_channels, n_frames, window].
    x = inputs[:, :, :, None]
    window = window[None, None, None, :]
    x_windowed = x * window
    # Collapse channel into batch size
    x_windowed = torch.reshape(
        x_windowed, (-1, x_windowed.shape[2], x_windowed.shape[3])
    )
    x = overlap_and_add(x_windowed, hop_size)
    # Reshape back to original shape.
    x = torch.reshape(x, (inputs.shape[0], inputs.shape[1], -1))

    # Trim the rise and fall of the first and last window.
    return x[:, :, hop_size:-hop_size]
