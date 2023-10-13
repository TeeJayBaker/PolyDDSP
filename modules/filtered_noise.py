"""
Filtered noise synthesiser
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import fftpack
import numpy as np

class FilteredNoise(nn.Module):
    """
    Filtered noise synthesiser

    Args:
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        device: Specify whether computed on cpu, cuda or mps

    Input: Filter coefficients of size (batch, frames, banks)
    Output: Filtered noise audio (batch, samples)
    """
    def __init__(self, frame_length = 64, attenuate_gain = 1e-2, device = 'mps'):
        super(FilteredNoise, self).__init__()
        
        self.frame_length = frame_length
        self.device = device
        self.attenuate_gain = attenuate_gain

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
    
    def fft_convolve(self, audio, impulse_response, padding = 'same', delay_compensation = -1):
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
        audio, impulse_response = audio.float(), impulse_response.float()

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
        ir_shape = list(impulse_response.size())
        batch_size_ir, n_ir_frames, ir_size = ir_shape

        # Validate that batch sizes match.
        if batch_size != batch_size_ir:
            raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                            'be the same.'.format(batch_size, batch_size_ir))

        # Cut audio into frames.
        frame_size = int(np.ceil(audio_size / n_ir_frames))
        hop_size = frame_size

        # Pad audio to match frame size (and match tf.signal.frame())
        pad_size = frame_size - abs(audio_size % hop_size)
        audio = F.pad(audio, (0, pad_size))
        audio_frames = audio.unfold(-1, frame_size, hop_size)

        # Check that number of frames match.
        n_audio_frames = int(audio_frames.shape[1])
        if n_audio_frames != n_ir_frames:
            raise ValueError(
                'Number of Audio frames ({}) and impulse response frames ({}) do not '
                'match. For small hop size = ceil(audio_size / n_ir_frames), '
                'number of impulse response frames must be a multiple of the audio '
                'size.'.format(n_audio_frames, n_ir_frames))

        # Pad and FFT the audio and impulse responses.
        fft_size = self.get_fft_size(frame_size, ir_size, power_of_2=True)
        audio_fft = torch.fft.rfft(audio_frames, fft_size)
        ir_fft = torch.fft.rfft(impulse_response, fft_size)

        # Multiply the FFTs (same as convolution in time).
        audio_ir_fft = torch.mul(audio_fft, ir_fft)

        # Take the IFFT to resynthesize audio.
        audio_frames_out = torch.fft.irfft(audio_ir_fft)
        audio_out = self.overlap_and_add(audio_frames_out, hop_size)

        # Crop and shift the output audio.
        return self.crop_and_compensate_delay(audio_out, audio_size, ir_size,
                                              delay_compensation, padding)

    def forward(self, z):
        """
        Compute LTI-FVR filter banks, and calculate time varying filtered noise via overlap-add.
        """

        batch_size, num_frames, num_banks = z['H'].shape

import tensorflow as tf



