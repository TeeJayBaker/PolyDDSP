"""Run Neutone's streaming and offline audio-to-pianoroll ONNX exports."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class NeutoneONNXModel:
    """Expose ONNX logits through the same interface as the Lightning model.

    Audio is passed to ONNX Runtime on the CPU; the selected execution provider
    handles any GPU transfers. Older exports without metadata use Neutone's
    original 44.1 kHz / 512-sample-hop timing.
    """

    input_device = "cpu"
    output_names = ("onset", "frame", "offset")

    def __init__(self, path: Path, device: str = "cpu", *, num_threads: int | None = None) -> None:
        import onnxruntime as ort

        target = torch.device(device)
        providers = ["CPUExecutionProvider"]
        if target.type == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise ValueError(
                    "ONNX CUDA inference requires onnxruntime-gpu; "
                    "install it or use --device cpu."
                )
            providers.insert(0, (
                "CUDAExecutionProvider", {"device_id": target.index or 0},
            ))
        elif target.type != "cpu":
            raise ValueError("Neutone ONNX inference supports --device cpu or cuda[:index].")

        # Loading by filename lets ORT resolve adjacent external weight files.
        session_kwargs = {}
        if num_threads is not None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
            session_kwargs["sess_options"] = options
        self.session = ort.InferenceSession(str(path), providers=providers, **session_kwargs)
        if target.type == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError(
                "ONNX Runtime could not initialize CUDA. Check its CUDA/cuDNN "
                "diagnostics above, or use --device cpu."
            )

        inputs = {item.name: item for item in self.session.get_inputs()}
        self.streaming = set(inputs) == {"input_audio", "cache"}
        if (set(inputs) not in ({"input_audio"}, {"input_audio", "cache"})
                or inputs["input_audio"].type != "tensor(float)"
                or len(inputs["input_audio"].shape) != 2):
            raise ValueError(
                "Expected a Neutone ONNX export with float32 input_audio shaped "
                "(batch, samples), and an optional streaming cache input."
            )
        available = {output.name for output in self.session.get_outputs()}
        if not set(self.output_names).issubset(available):
            raise ValueError("Neutone ONNX export must provide onset, frame, and offset outputs.")

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.spec = SimpleNamespace(
            sample_rate=int(metadata.get("sample_rate", 44_100)),
            hop_length=int(metadata.get("hop_length", 512)),
        )
        if self.spec.sample_rate <= 0 or self.spec.hop_length <= 0:
            raise ValueError("ONNX sample_rate and hop_length metadata must be positive.")
        # cuDNN RNN descriptors allow at most 65,535 sequence steps. Leave
        # room for the offline export's internal padding (16 extra frames in
        # the current export), instead of accepting audio right at that limit.
        self.max_input_samples = (
            65_000 * self.spec.hop_length
            if target.type == "cuda" and not self.streaming else None
        )
        # Current offline exports compensate reporting delay inside the graph.
        # Applying it again would move all notes earlier by the same amount.
        self.target_shift = (
            0 if metadata.get("target_shift_baked", "0") == "1"
            else int(metadata.get("target_shift_frames", 0))
        )
        self._delays = [int(value) for value in metadata.get("delays_frames", "").split(",") if value]
        if self.streaming:
            cache = inputs["cache"]
            self._cache_shape = tuple(cache.shape)
            if (inputs["input_audio"].shape != [1, self.spec.hop_length]
                    or cache.type != "tensor(float)" or len(self._cache_shape) != 2
                    or self._cache_shape[0] != 1
                    or not isinstance(self._cache_shape[1], int) or self._cache_shape[1] <= 0
                    or "next_cache" not in available):
                raise ValueError(
                    "Streaming Neutone ONNX requires one hop of audio per call, "
                    "a fixed (1, cache_size) float32 cache, and a next_cache output."
                )
            reporting_delay = self._delays[0] if self._delays else self.target_shift
            self._stream_shift = int(metadata.get(
                "latency_frames", int(metadata.get("lookahead_frames", 0)) + reporting_delay,
            ))
            if self._stream_shift < 0:
                raise ValueError("ONNX latency_frames metadata must be nonnegative.")
            # Flush and realign streaming predictions here, including the tail.
            self.target_shift = 0

    def _rolls(self, values: list[np.ndarray], batch_size: int) -> dict[str, np.ndarray]:
        outputs = dict(zip(self.output_names, values))
        if outputs["onset"].ndim == 4:
            if not self._delays or any(value.shape[1] != len(self._delays) for value in outputs.values()):
                raise ValueError("Multi-delay ONNX outputs require matching delays_frames metadata.")
            # Match the checkpoint pipeline's default: the first/lowest delay.
            outputs = {key: value[:, 0] for key, value in outputs.items()}
        shape = outputs["onset"].shape
        if (len(shape) != 3 or shape[0] != batch_size or shape[1] != 88
                or any(value.shape != shape for value in outputs.values())):
            raise ValueError("Expected matching Neutone ONNX logits shaped (batch, 88, frames).")
        return outputs

    def _run_streaming(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        if audio.shape[0] != 1:
            raise ValueError("Streaming Neutone ONNX processes one audio file at a time.")
        hop = self.spec.hop_length
        frames = (audio.shape[-1] + hop - 1) // hop
        steps = frames + self._stream_shift
        audio = np.pad(audio, ((0, 0), (0, steps * hop - audio.shape[-1])))
        # State belongs to this file, never to the reusable model instance.
        cache = np.zeros(self._cache_shape, dtype=np.float32)
        outputs = {key: np.empty((1, 88, steps), dtype=np.float32) for key in self.output_names}
        for index in range(steps):
            values = self.session.run(
                [*self.output_names, "next_cache"],
                {"input_audio": audio[:, index * hop:(index + 1) * hop], "cache": cache},
            )
            cache = values[-1]
            rolls = self._rolls(values[:-1], batch_size=1)
            for key, value in rolls.items():
                if value.shape[-1] != 1:
                    raise ValueError("Streaming Neutone ONNX must produce one frame per audio hop.")
                outputs[key][..., index:index + 1] = value
        return {key: value[..., self._stream_shift:] for key, value in outputs.items()}

    def __call__(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        audio_np = audio.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if self.streaming:
            outputs = self._run_streaming(audio_np)
        else:
            values = self.session.run(list(self.output_names), {"input_audio": audio_np})
            outputs = self._rolls(values, audio.shape[0])
        return {key: torch.from_numpy(value) for key, value in outputs.items()}
