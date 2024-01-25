"""
Pitch prediction module, porting basic_pitch to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import nnAudio.features.cqt as nnAudio
import math
import utils
from typing import Optional, Tuple, List

MIDI_OFFSET = 21

class harmonic_stacking(nn.Module):
    """
    Harmonic stacking layer

    Input: (batch, freq_bins, time_frames)

    Args:
        bins_per_semitone: The number of bins per semitone in the CQT
        harmonics: The list of harmonics to stack
        n_output_freqs: The number of output frequencies to return in each harmonic layer
        
    Returns: 
        (batch, n_harmonics, out_freq_bins, time_frames)
    """

    def __init__(self, bins_per_semitone, harmonics, n_output_freqs):
        super(harmonic_stacking, self).__init__()

        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs
        self.shifts = [int(np.round(12 * self.bins_per_semitone * np.log2(float(h)))) for h in self.harmonics]

    def forward(self, x):
        torch._assert(len(x.shape) == 3, "x must be (batch, freq_bins, time_frames)")
        channels = []
        for shift in self.shifts:
            if shift == 0:
                channels.append(x)
            elif shift > 0:
                channels.append(F.pad(x[:, shift:, :], (0, 0, 0, shift)))
            elif shift < 0:
                channels.append(F.pad(x[:, :shift, :], (0, 0, -shift, 0)))
            else:
                raise ValueError("shift must be non-zero")
        x = torch.stack(channels, dim=1)
        x = x[:, :, :self.n_output_freqs, :]
        return x
    
class Conv2dSame(torch.nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    
def constrain_frequency(onsets: torch.Tensor,
                        frames: torch.Tensor, 
                        max_freq: Optional[float], 
                        min_freq: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constrain the frequency range of the pitch predictions by zeroing out bins outside the range

    Args:
        onsets: The onset predictions (batch, freq, time_frames)
        frames: The frame predictions (batch, freq, time_frames)
        max_freq: The maximum frequency to allow
        min_freq: The minimum frequency to allow

    Returns:
        The constrained onset and frame predictions
    """
    # check inputs are batched
    torch._assert(len(onsets.shape) == 3, "onsets must be (batch, freq, time_frames)")
    torch._assert(len(frames.shape) == 3, "frames must be (batch, freq, time_frames)")
    
    if max_freq is not None:
        max_freq_idx = int(np.round(utils.hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:, :].zero_()
        frames[:, max_freq_idx:, :].zero_()
    if min_freq is not None:
        min_freq_idx = int(np.round(utils.hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx, :].zero_()
        frames[:, :min_freq_idx, :].zero_()

    return onsets, frames

def get_infered_onsets(onsets: torch.Tensor,
                       frames: torch.Tensor,
                       n_diff: int = 2) -> torch.Tensor:
    """
    Infer onsets from large changes in frame amplutude

    Args:
        onsets: The onset predictions (batch, freq, time_frames)
        frames: The frame predictions (batch, freq, time_frames)
        n_diff: DDifferences used to detect onsets

    Returns:
        The maximum between the predicted onsets and its differences
    """
    # check inputs are batched
    torch._assert(len(onsets.shape) == 3, "onsets must be (batch, freq, time_frames)")
    torch._assert(len(frames.shape) == 3, "frames must be (batch, freq, time_frames)")

    diffs = []
    for n in range(1, n_diff + 1):
        # Use PyTorch's efficient padding and slicing
        frames_padded = torch.nn.functional.pad(frames, (0, 0, n, 0))
        diffs.append(frames_padded[:, n:, :] - frames_padded[:, :-n, :])

    frame_diff = torch.min(torch.stack(diffs), dim=0)[0]
    frame_diff = torch.clamp(frame_diff, min=0)  # Replaces negative values with 0
    frame_diff[:, :n_diff, :].zero_()  # Set the first n_diff frames to 0

    # Rescale to have the same max as onsets
    frame_diff = frame_diff * torch.max(onsets) / torch.max(frame_diff)

    # Use the max of the predicted onsets and the differences
    max_onsets_diff = torch.max(onsets, frame_diff)

    return max_onsets_diff

def argrelmax(x: torch.Tensor, width: int = 3) -> torch.Tensor:
    """
    Emulate scipy.signal.argrelmax in torch

    Args:
        x: The input tensor
        axis: The axis along which to find local maxima

    Returns:
        The indices of the local maxima
    """
    # check inputs are batched
    torch._assert(len(x.shape) == 3, "x must be (batch, freq, time_frames)")
    #print(x)
    window_maxima = F.max_pool1d_with_indices(x, width, 1, padding = width//2)[1]
    print(window_maxima)
    candidates = window_maxima.unique()
    print(candidates)
    peaks = candidates[(window_maxima[candidates]==candidates).nonzero()]

    return peaks


class basic_pitch(nn.Module):
    """
    Port of basic_pitch pitch prediction to pytorch

    input: (batch, samples)
    returns: {"onset": (batch, freq_bins, time_frames),
              "contour": (batch, freq_bins, time_frames),
              "note": (batch, freq_bins, time_frames)}
    """

    def __init__(self,
                sr=16000,
                hop_length=256,
                annotation_semitones=88,
                annotation_base=27.5,
                n_harmonics=8,
                n_filters_contour=32,
                n_filters_onsets=32,
                n_filters_notes=32,
                no_contours=False,
                contour_bins_per_semitone=3,
                device='mps'):
        super(basic_pitch, self).__init__()

        self.sr = sr
        self.hop_length = hop_length
        self.annotation_semitones = annotation_semitones
        self.annotation_base = annotation_base
        self.n_harmonics = n_harmonics
        self.n_filters_contour = n_filters_contour
        self.n_filters_onsets = n_filters_onsets
        self.n_filters_notes = n_filters_notes
        self.no_contours = no_contours
        self.contour_bins_per_semitone = contour_bins_per_semitone
        self.n_freq_bins_contour = annotation_semitones * contour_bins_per_semitone
        self.device = device

        max_semitones = int(np.floor(12.0 * np.log2(0.5 * self.sr / self.annotation_base)))

        n_semitones = np.min(
            [
                int(np.ceil(12.0 * np.log2(self.n_harmonics)) + self.annotation_semitones),
                max_semitones,
            ]
        )

        self.cqt = nnAudio.CQT2010v2(sr = self.sr,
                          fmin = self.annotation_base,
                          hop_length = self.hop_length,
                          n_bins = n_semitones * self.contour_bins_per_semitone,
                          bins_per_octave = 12 * self.contour_bins_per_semitone,
                          verbose=False)
        
        self.contour_1 = nn.Sequential(
            nn.Conv2d(self.n_harmonics, self.n_filters_contour, (5, 5), padding='same'),
            nn.BatchNorm2d(self.n_filters_contour),
            nn.ReLU(),
            nn.Conv2d(self.n_filters_contour, 8, (3 * 13, 3), padding='same'),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.contour_2 = nn.Sequential(
            nn.Conv2d(8, 1, (5, 5), padding='same'),
            nn.Sigmoid()
        )

        self.note_1 = nn.Sequential(
            Conv2dSame(1, self.n_filters_notes, (7, 7), (3, 1)),
            nn.ReLU(),
            nn.Conv2d(self.n_filters_notes, 1, (3, 7), padding='same'),
            nn.Sigmoid()
        )

        self.onset_1 = nn.Sequential(
            Conv2dSame(self.n_harmonics, self.n_filters_onsets, (5, 5), (3, 1)),
            nn.BatchNorm2d(self.n_filters_onsets),
            nn.ReLU()
        )

        self.onset_2 = nn.Sequential(
            nn.Conv2d(self.n_filters_onsets + 1, 1, (5, 5), padding='same'),
            nn.Sigmoid()
        )

        
    def normalised_to_db(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert spectrogram to dB and normalise

        Args:
            audio: The spectrogram input. (batch, 1, freq_bins, time_frames) 
                or (batch, freq_bins, time_frames)

        Returns: 
            the spectogram in dB in the same shape as the input
        """
        power = torch.square(audio)
        log_power = 10.0 * torch.log10(power + 1e-10)

        log_power_min = torch.min(log_power, keepdim=True, dim = -2)[0]
        log_power_min = torch.min(log_power_min, keepdim=True, dim = -1)[0]
        log_power_offset = log_power - log_power_min
        log_power_offset_max = torch.max(log_power_offset, keepdim=True, dim=-2)[0]
        log_power_offset_max = torch.max(log_power_offset_max, keepdim=True, dim=-1)[0]    

        log_power_normalised = log_power_offset / log_power_offset_max
        return log_power_normalised

    

    def get_cqt(self, audio: torch.Tensor, use_batchnorm: bool) -> torch.Tensor:
        """
        Compute CQT from audio

        Args:
            audio: The audio input. (batch, samples)
            n_harmonics: The number of harmonics to capture above the maximum output frequency.
                Used to calculate the number of semitones for the CQT.
            use_batchnorm: If True, applies batch normalization after computing the CQT


        Returns: 
            the log-normalised CQT of audio (batch, freq_bins, time_frames)
        """

        torch._assert(len(audio.shape) == 2, "audio has multiple channels, only mono is supported")
        x = self.cqt(audio)
        x = self.normalised_to_db(x)

        if use_batchnorm:
            x = torch.unsqueeze(x, 1)
            x = nn.BatchNorm2d(1).to(self.device)(x)
            x = torch.squeeze(x, 1)

        return x
    
    def forward(self, x):
        x = self.get_cqt(x, use_batchnorm=True)

        if self.n_harmonics > 1:
            x = harmonic_stacking(self.contour_bins_per_semitone,
                                  [0.5] + list(range(1, self.n_harmonics)),
                                  self.n_freq_bins_contour)(x)
        else:
            x = harmonic_stacking(self.contour_bins_per_semitone,
                                  [1],
                                  self.n_freq_bins_contour)(x)
        
        # contour layers         
        x_contours = self.contour_1(x)

        if not self.no_contours:
            x_contours = self.contour_2(x_contours)
            x_contours = torch.squeeze(x_contours, 1)
            x_contours_reduced = torch.unsqueeze(x_contours, 1) 
        else:
            x_contours_reduced = x_contours

        # notes layers
        x_notes_pre = self.note_1(x_contours_reduced)
        x_notes = torch.squeeze(x_notes_pre, 1)

        # onsets layers
        x_onset = self.onset_1(x)
        x_onset = torch.cat([x_notes_pre, x_onset], dim=1)
        x_onset = self.onset_2(x_onset)
        x_onset = torch.squeeze(x_onset, 1)

        return {"onset": x_onset, "contour": x_contours, "note": x_notes}
    

def output_to_notes_polyphonic(frames: torch.Tensor,
                               onsets: torch.Tensor,
                               onset_thresh: float,
                               frame_thresh: float,
                               min_note_len: int,
                               infer_onsets: bool,
                               max_freq: Optional[float],
                               min_freq: Optional[float],
                               melodia_trick: bool = True,
                               energy_tol: int = 11) -> List[Tuple[int, int, int, float]]:
    """
    Convert pitch predictions to note predictions

    Args:
        frames: The frame predictions (freq_bins, time_frames)
        onsets: The onset predictions (freq_bins, time_frames)
        onset_thresh: The threshold for onset detection
        frame_thresh: The threshold for frame detection
        min_note_len: The minimum number of frames for a note to be valid
        infer_onsets: If True, infer onsets from large changes in frame amplitude
        max_freq: The maximum frequency to allow
        min_freq: The minimum frequency to allow
        melodia_trick: If True, use the Melodia trick to remove spurious notes
        energy_tol: The energy tolerance for the Melodia trick

    Returns:
        A list of notes in the format (start_time, end_time, pitch, velocity)
    """

    n_frames = frames.shape[-1]

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)
    
    peak_thresh_mat = torch.zeros_like(onsets)

    raise NotImplementedError

def test_argrelmax():
    # Create a random numpy array
    data = torch.rand(1,5,100)  # Shape: (100,)

    # Call your argrelmax function
    maxima = argrelmax(data)

    # Print the indices of the relative maxima
    print(maxima)

# Call the test function
test_argrelmax()