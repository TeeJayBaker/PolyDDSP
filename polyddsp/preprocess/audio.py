"""Audio utilities shared by transcription backends."""
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF


def load_resample(path: Path, target_sr: int) -> torch.Tensor:
    """Read an audio file as mono float32 and resample to ``target_sr``."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    audio = torch.from_numpy(wav)
    if sr != target_sr:
        audio = AF.resample(audio, orig_freq=sr, new_freq=target_sr)
    return audio
