"""
Utility functions for basic_pitch
"""

import numpy as np
import torch
import torch.nn.functional as F


def hz_to_midi(frequency: float) -> float:
    """
    Convert a frequency in Hz to a MIDI note number
    """
    return 69 + 12 * np.log2(frequency / 440.0)


def midi_to_hz(midi_note: int) -> float:
    """
    Convert a MIDI note number to a frequency in Hz
    """
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def tensor_midi_to_hz(midi_note: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of MIDI note numbers to a tensor of frequencies in Hz
    """
    return 440.0 * 2 ** ((midi_note - 69) / 12)


def unravel_index(index: int, shape: tuple) -> tuple:
    """
    Convert a flat index into a multi-dimensional index
    """
    unravelled = []
    for dim in reversed(shape):
        unravelled.append(index % dim)
        index = index // dim
    return tuple(reversed(unravelled))


def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    pad_size = 0
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length - frames_overlap) % np.abs(
            frame_length - frames_overlap
        )
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames, pad_size
