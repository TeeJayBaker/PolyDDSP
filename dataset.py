"""
Dataset class for loading audio data
"""

import torchaudio
from torch.utils.data import Dataset
import numpy as np


class AudioDataset(Dataset):
    """
    Dataset class for loading audio data

    Args:
        paths: paths to audio files
        sr: sample rate of audio
        duration: duration of audio
        random: whether to random sample audio
    """

    def __init__(
        self,
        paths: list[str],
        sr: int = 22050,
        duration: float = 10.0,
        random: bool = True,
    ):
        super(AudioDataset, self).__init__()

        self.paths = paths
        self.sr = sr
        self.duration = duration
        self.random = random

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        audio, old_sr = torchaudio.load(self.paths[idx], normalize=True)
        audio = audio[0]
        audio = audio.unsqueeze(0)
        audio = torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=self.sr)(
            audio
        )
        if self.random:
            start = np.random.randint(0, audio.shape[-1] - int(self.duration * self.sr))
            audio = audio[:, start : start + int(self.duration * self.sr)]
        else:
            audio = audio[:, : int(self.duration * self.sr)]

        return audio
