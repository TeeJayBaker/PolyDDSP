"""
Filtered noise synthesiser
"""

import torch
import torch.nn as nn
import modules.operations as ops


class FilteredNoise(nn.Module):
    """
    Filtered noise synthesiser

    Args:
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        initial_bias: Initial bias for the sigmoid function
        window_size: Size of the window to apply in the time domain for FIR filter
        device: Specify whether computed on cpu, cuda or mps

    Input: Filter coefficients of size (batch, frames, banks)
    Output: Filtered noise audio (batch, samples)
    """

    def __init__(
        self,
        frame_length: int = 64,
        attenuate_gain: float = 1e-2,
        initial_bias: float = -5.0,
        window_size: int = 257,
        device: str = "mps",
    ):
        super(FilteredNoise, self).__init__()

        self.initial_bias = initial_bias
        self.window_size = window_size
        self.frame_length = frame_length
        self.attenuate_gain = attenuate_gain
        self.device = device
        self.to(device)

    def apply_window_to_impulse_response(
        self, impulse_response: torch.Tensor, window_size: int = 0, causal: bool = False
    ) -> torch.Tensor:
        """
        Apply window to impulse response and put it in causal form.

        Args:
            impulse_response: Impulse response frames to apply window to [batch, n_frames, ir_size].
            window_size: Size of window to apply in time domain. If less than 1, defaults to impulse response size.
            causal: impulse response is in causal form (central peak)

        Returns:
            impulse_response: Windowed impulse response in causal form, with last
                dimension cropped to window_size if window_size is greater than 0 and less
                than ir_size.
        """
        impulse_response = impulse_response.float()

        # if IR is causal, convert to zero-phase
        if causal:
            impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

        # Get a window for better time/frequency resolution than rectangular.
        # Window defaults to IR size, cannot be bigger.
        ir_size = int(impulse_response.shape[-1])
        if (window_size <= 0) or (window_size > ir_size):
            window_size = ir_size
        window = torch.hann_window(window_size, device=impulse_response.device)

        # Zero pad the window and put in in zero-phase form.
        padding = ir_size - window_size
        if padding > 0:
            half_idx = (window_size + 1) // 2
            window = torch.cat(
                (window[half_idx:], torch.zeros(padding), window[:half_idx]), dim=0
            )
        else:
            window = torch.fft.fftshift(window, dim=-1)

        # Apply the window, to get new IR (both in zero-phase form).
        window = torch.broadcast_to(window, impulse_response.shape)
        impulse_response = torch.mul(window, impulse_response.real)

        # Put IR in causal form and trim zero padding.
        if padding > 0:
            first_half_start = (ir_size - (half_idx - 1)) + 1
            second_half_end = half_idx + 1
            impulse_response = torch.cat(
                [
                    impulse_response[..., first_half_start:],
                    impulse_response[..., :second_half_end],
                ],
                axis=-1,
            )
        else:
            impulse_response = torch.fft.fftshift(impulse_response, dim=-1)

        return impulse_response

    def frequency_impulse_response(
        self, magnitudes: torch.Tensor, window_size: int = 0
    ) -> torch.Tensor:
        """Get windowed impulse responses using the frequency sampling method.

        Follows the approach in:
        https://ccrma.stanford.edu/~jos/sasp/Windowing_Desired_Impulse_Response.html

        Args:
            magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
                n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
                last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
                f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
                audio into equally sized frames to match frames in magnitudes.
            window_size: Size of the window to apply in the time domain. If window_size
                is less than 1, it defaults to the impulse_response size.

        Returns:
            impulse_response: Time-domain FIR filter of shape
                [batch, frames, window_size] or [batch, window_size].

        Raises:
            ValueError: If window size is larger than fft size.
        """

        # Get the IR (zero-phase form).
        # magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
        impulse_response = torch.fft.irfft(magnitudes)

        # Window and put in causal form.
        impulse_response = self.apply_window_to_impulse_response(
            impulse_response, window_size
        )

        return impulse_response

    def frequency_filter(
        self,
        audio: torch.Tensor,
        magnitudes: torch.Tensor,
        window_size: int = 0,
        padding: str = "same",
    ) -> torch.Tensor:
        """Filter audio with a finite impulse response filter.

        Args:
            audio: Input audio. Tensor of shape [batch, audio_timesteps].
            magnitudes: Frequency transfer curve. Float32 Tensor of shape [batch,
                n_frames, n_frequencies] or [batch, n_frequencies]. The frequencies of the
                last dimension are ordered as [0, f_nyqist / (n_frequencies -1), ...,
                f_nyquist], where f_nyquist is (sample_rate / 2). Automatically splits the
                audio into equally sized frames to match frames in magnitudes.
            window_size: Size of the window to apply in the time domain. If window_size
                is less than 1, it is set as the default (n_frequencies).
            padding: Either 'valid' or 'same'. For 'same' the final output to be the
                same size as the input audio (audio_timesteps). For 'valid' the audio is
                extended to include the tail of the impulse response (audio_timesteps +
                window_size - 1).

        Returns:
            Filtered audio. Tensor of shape
                [batch, audio_timesteps + window_size - 1] ('valid' padding) or shape
                [batch, audio_timesteps] ('same' padding).
        """
        impulse_response = self.frequency_impulse_response(
            magnitudes, window_size=window_size
        )
        return ops.fft_convolve(audio, impulse_response, padding=padding)

    def forward(self, filter_coeff: torch.Tensor) -> torch.Tensor:
        """
        Compute LTI-FVR filter banks, and calculate time varying filtered noise via overlap-add.
        """

        batch_size, _, num_banks, num_frames = filter_coeff.shape
        filter_coeff = filter_coeff.mean(dim=1).permute(0, 2, 1)
        magnitudes = ops.exp_sigmoid(filter_coeff + self.initial_bias)

        noise = torch.empty(
            batch_size, num_frames * self.frame_length, device=self.device
        ).uniform_(-1, 1)
        return (
            self.frequency_filter(noise, magnitudes, window_size=self.window_size)
            * self.attenuate_gain
        )
