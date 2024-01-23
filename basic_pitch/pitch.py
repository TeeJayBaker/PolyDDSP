"""
Pitch prediction module, porting basic_pitch to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import nnAudio.features.cqt as cqt
import math

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

class basic_pitch(nn.Module):
    """
    Port of basic_pitch pitch prediction to pytorch
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

        max_semitones = int(np.floor(12.0 * np.log2(0.5 * self.sr / self.annotation_base)))

        n_semitones = np.min(
            [
                int(np.ceil(12.0 * np.log2(self.n_harmonics)) + self.annotation_semitones),
                max_semitones,
            ]
        )

        torch._assert(len(audio.shape) == 2, "audio has multiple channels, only mono is supported")
        x = cqt.CQT2010v2(sr = self.sr,
                          fmin = self.annotation_base,
                          hop_length = self.hop_length,
                          n_bins = n_semitones * self.contour_bins_per_semitone,
                          bins_per_octave = 12 * self.contour_bins_per_semitone,
                          verbose=False).to(self.device)(audio)
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
            input_size = self.n_harmonics
        else:
            x = harmonic_stacking(self.contour_bins_per_semitone,
                                  [1],
                                  self.n_freq_bins_contour)(x)
            input_size = 1
        
        # contour layers
        x_contours = nn.Conv2d(input_size, self.n_filters_contour, (5, 5), padding='same')(x)
        x_contours = nn.BatchNorm2d(self.n_filters_contour)(x_contours)
        x_contours = nn.ReLU()(x_contours)

        x_contours = nn.Conv2d(self.n_filters_contour, 8, (3, 3 * 13), padding='same')(x_contours)    
        x_contours = nn.BatchNorm2d(8)(x_contours)
        x_contours = nn.ReLU()(x_contours)

        if not self.no_contours:
            x_contours = nn.Conv2d(8, 1, (5, 5), padding='same')(x_contours)
            x_contours = nn.Sigmoid()(x_contours)
            x_contours = torch.squeeze(x_contours, 1)
            x_contours_reduced = torch.unsqueeze(x_contours, 1) 
        else:
            x_contours_reduced = x_contours

        x_contours_reduced = Conv2dSame(1, self.n_filters_notes, (7, 7), (1, 3))(x_contours_reduced)
        x_contours_reduced = nn.ReLU()(x_contours_reduced)

        # notes layers
        x_notes_pre = nn.Conv2d(self.n_filters_notes, 1, (7, 3), padding='same')(x_contours_reduced)   
        x_notes_pre = nn.Sigmoid()(x_notes_pre)
        x_notes = torch.squeeze(x_notes_pre, 1)

        # onsets layers
        x_onset = Conv2dSame(input_size, self.n_filters_onsets, (5, 5), (1, 3))(x)
        x_onset = nn.BatchNorm2d(self.n_filters_onsets)(x_onset)
        x_onset = nn.ReLU()(x_onset)

        x_onset = torch.cat([x_notes_pre, x_onset], dim=1)
        x_onset = nn.Conv2d(self.n_filters_onsets + 1, 1, (5, 5), padding='same')(x_onset)
        x_onset = torch.squeeze(x_onset, 1)

        return {"onset": x_onset, "contour": x_contours, "note": x_notes}
    

        

def test_basic_pitch():
    # Create a random tensor for audio
    audio = torch.rand((1, 16000)).to('mps')  # 1 second of audio at 16000 Hz

    # Instantiate a basic_pitch object
    pitch_detector = basic_pitch(sr=16000, hop_length=512, n_harmonics=8, device='mps').to('mps')

    # Call the forward method
    output = pitch_detector(audio)

    # Print the shape of the output
    print(output['onset'].shape)
    print(output['contour'].shape)
    print(output['note'].shape)

# Call the test function
test_basic_pitch()