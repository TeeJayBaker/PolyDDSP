"""
Utility functions for basic_pitch
"""

import numpy as np

def hz_to_midi(frequency: float) -> float:
    """
    Convert a frequency in Hz to a MIDI note number
    """
    return 69 + 12 * np.log2(frequency / 440.0)