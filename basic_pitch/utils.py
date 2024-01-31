"""
Utility functions for basic_pitch
"""

import numpy as np

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