"""
Pitch prediction module, porting basic_pitch to pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import nnAudio.features.cqt as cqt

class harmonic_stacking(nn.Module):
    """
    Harmonic stacking layer

    Input: (batch, freq_bins, time_frames)

    Args:
        bins_per_semitone: The number of bins per semitone in the CQT
        harmonics: The list of harmonics to stack
        n_output_freqs: The number of output frequencies to return in each harmonic layer
        
    Returns: 
        (batch, out_freq_bins, n_harmonics, time_frames)
    """

    def __init__(self, bins_per_semitone, harmonics, n_output_freqs):
        super(harmonic_stacking, self).__init__()

        self.bins_per_semitone = bins_per_semitone
        self.harmonics = harmonics
        self.n_output_freqs = n_output_freqs
        self.shifts = [int(torch.round(12 * self.bins_per_semitone * np.log2(float(h)))) for h in self.harmonics]

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
        x = torch.stack(channels, dim=2)
        x = x[:, :, :self.n_output_freqs, :]
        return x

class basic_pitch(nn.Module):
    """
    Port of basic_pitch pitch prediction to pytorch
    """

    def __init__(self,
                sr=16000,
                hop_length=512,
                max_semitones=72,
                annotation_semitones=0,
                annotation_base=55.0,
                bins_per_semitone=1,
                n_harmonics=8,
                n_filters_contour=32,
                n_filters_onsets=32,
                n_filters_notes=32,
                no_contours=False,
                contour_bins_per_semitone=1,
                n_freq_bins_contour=360,
                device='mps'):
        super(basic_pitch, self).__init__()

        self.sr = sr
        self.hop_length = hop_length
        self.max_semitones = max_semitones
        self.annotation_semitones = annotation_semitones
        self.annotation_base = annotation_base
        self.bins_per_semitone = bins_per_semitone
        self.n_harmonics = n_harmonics
        self.n_filters_contour = n_filters_contour
        self.n_filters_onsets = n_filters_onsets
        self.n_filters_notes = n_filters_notes
        self.no_contours = no_contours
        self.contour_bins_per_semitone = contour_bins_per_semitone
        self.n_freq_bins_contour = n_freq_bins_contour
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
        n_semitones = np.min(
            [
                int(np.ceil(12.0 * np.log2(self.n_harmonics)) + self.annotation_semitones),
                self.max_semitones,
            ]
        )
        torch._assert(len(audio.shape) == 2, "audio has multiple channels, only mono is supported")
        x = cqt.CQT2010v2(sr = self.sr,
                          fmin = self.annotation_base,
                          hop_length = self.hop_length,
                          n_bins = n_semitones * self.bins_per_semitone,
                          bins_per_octave = 12 * self.bins_per_semitone,
                          verbose=False)(audio)
        x = self.normalised_to_db(x)

        if use_batchnorm:
            x = torch.unsqueeze(x, 1)
            x = nn.BatchNorm2d(1)(x)
            x = torch.squeeze(x, 1)

        return x
    
    def forward(self, x):
        if self.num_harmonics > 1:
            x = self.harmonic_stacking(self.contour_bins_per_semitone,
                                       [0.5] + list(range(1, self.num_harmonics)),
                                       self.n_freq_bins_contour)(x)
        else:
            x = self.harmonic_stacking(self.contour_bins_per_semitone,
                                       [1],
                                       self.n_freq_bins_contour)(x)
            
        # contour layers
            
        # notes layers
            
        # onsets layers
        raise NotImplementedError
        

