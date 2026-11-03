"""Small bounded playback utilities; no model work runs in the audio callback."""
from __future__ import annotations

from threading import Lock

import numpy as np


class LinearOutputResampler:
    """Incremental same-rate/upsampling with one-sample history and a rational clock.

    Linear interpolation is a lightweight prototype output converter, not a
    high-quality bandlimited resampler. Downsampling is intentionally rejected.
    """

    def __init__(self, input_rate: int, output_rate: int) -> None:
        if input_rate < 1 or output_rate < input_rate:
            raise ValueError("output_rate must be >= the positive model sample rate")
        self.input_rate, self.output_rate = input_rate, output_rate
        self._history = np.empty(0, dtype=np.float32)
        self._received = self._next_output = 0
        self._finished = False

    def process(self, audio: np.ndarray, *, final: bool = False) -> np.ndarray:
        if self._finished:
            raise RuntimeError("resampler is finished")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be mono")
        start = self._received - len(self._history)
        buffered = np.concatenate([self._history, audio])
        self._received += len(audio)
        self._finished = final
        if not self._received:
            return np.empty(0, dtype=np.float32)
        if final:
            end = (self._received * self.output_rate + self.input_rate - 1) // self.input_rate
        else:
            end = (self._received - 1) * self.output_rate // self.input_rate + 1
        indices = np.arange(self._next_output, end, dtype=np.int64)
        numerator = indices * self.input_rate
        left = numerator // self.output_rate - start
        fraction = (numerator % self.output_rate) / self.output_rate
        right = np.minimum(left + 1, len(buffered) - 1)
        output = buffered[left] * (1 - fraction) + buffered[right] * fraction
        self._history = buffered[-1:].copy()
        self._next_output = end
        return output.astype(np.float32)


class AudioRingBuffer:
    """Fixed-capacity mono queue; the callback never waits for the writer lock."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._buffer = np.empty(capacity, dtype=np.float32)
        self._read = self._write = self._count = 0
        self._lock = Lock()
        self.underruns = 0

    @property
    def free(self) -> int:
        with self._lock:
            return len(self._buffer) - self._count

    def write(self, audio: np.ndarray) -> int:
        with self._lock:
            count = min(len(audio), len(self._buffer) - self._count)
            first = min(count, len(self._buffer) - self._write)
            self._buffer[self._write:self._write + first] = audio[:first]
            self._buffer[:count - first] = audio[first:count]
            self._write = (self._write + count) % len(self._buffer)
            self._count += count
            return count

    def read_into(self, output: np.ndarray) -> None:
        output.fill(0)
        if not self._lock.acquire(blocking=False):
            self.underruns += 1
            return
        try:
            count = min(len(output), self._count)
            first = min(count, len(self._buffer) - self._read)
            output[:first] = self._buffer[self._read:self._read + first]
            output[first:count] = self._buffer[:count - first]
            self._read = (self._read + count) % len(self._buffer)
            self._count -= count
            if count < len(output):
                self.underruns += 1
        finally:
            self._lock.release()
