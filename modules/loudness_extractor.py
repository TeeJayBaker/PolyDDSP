"""
Extractor function for loudness envelopes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class LoudnessExtractor(nn.Module):
    """
    Loudness envelope extractor
    
    Args:
        sr: input audio sample rate
        frame_length: length of each frame window
        attenuate_gain: gain multiplier applied at end of generation
        device: Specify whether computed on cpu, cuda or mps

    Input: Audio input of size (batch, samples)
    Output: Loudness envelopes of size (batch, frames)
    """
    def __init__(self,
                sr = 16000,
                frame_length = 64,
                attenuate_gain = 2.,
                device = 'mps'):
    
        super(LoudnessExtractor, self).__init__()

        self.sr = sr
        self.frame_length = frame_length
        self.n_fft = self.frame_length # * 5
        self.device = device
        self.attenuate_gain = attenuate_gain
        self.smoothing_window = nn.Parameter(torch.hann_window(self.n_fft, dtype = torch.float32), 
                                             requires_grad = False).to(self.device)

    def A_weighting(self, frequencies, min_db=-45):
        """
        Calculate A-weighting in Decibel scale
        mirrors the librosa function of the same name

        Args:
            frequencies: tensor of frequencies to return weight
            min_db: minimum decibel weight to avoid exp/log errors
        
        Returns: Decibel weights for each frequency bin
        """

        f_sq = frequencies ** 2.0
        const = torch.tensor([12194.217, 20.598997, 107.65265, 737.86223], 
                             dtype=torch.float32).to(self.device) ** 2.0
        weights = 2.0 + 20.0 * (
                torch.log10(const[0])
                + 2 * torch.log10(f_sq)
                - torch.log10(f_sq + const[0])
                - torch.log10(f_sq + const[1])
                - 0.5 * torch.log10(f_sq + const[2])
                - 0.5 * torch.log10(f_sq + const[3])
                )       
        
        if min_db is None:
            return weights
        else:
            return torch.maximum(torch.tensor([min_db], dtype = torch.float32).to(self.device), weights)
        
    def forward(self, audio):
        """
        Compute loudness envelopes for audio input
        """

        padded_audio = F.pad(audio, (self.frame_length // 2, self.frame_length // 2))
        # sliced_audio = padded_audio.unfold(1, self.n_fft, self.frame_length)
        # sliced_windowed_audio = sliced_audio * self.smoothing_window


        # compute FFT step
        s = torch.stft(padded_audio, 
                                n_fft=self.frame_length, 
                                window=torch.hann_window(self.frame_length), 
                                center=False, 
                                return_complex=True)
        
        amplitude = torch.abs(s)
        power = amplitude ** 2

        frequencies = torch.fft.rfftfreq(self.n_fft, 1 / self.sr)
        a_weighting = self.A_weighting(frequencies).unsqueeze(0).unsqueeze(0)

        weighting = 10 ** (a_weighting/10)
        # print(weighting)
        # print(power)
        power = power.transpose(-1,-2) * weighting

        avg_power = torch.mean(power, -1)
        loudness = torchaudio.functional.amplitude_to_DB(avg_power, 
                                                         multiplier=10, 
                                                         amin=1e-4, 
                                                         db_multiplier=10)

        return loudness