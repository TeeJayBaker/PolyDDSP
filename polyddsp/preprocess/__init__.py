"""Pitch-cache preprocessing backends and stable public API."""
from polyddsp.preprocess.basic_pitch import cache_suffix as bp_cache_suffix
from polyddsp.preprocess.basic_pitch import precompute_one as precompute_one_bp
from polyddsp.preprocess.cache import cache_path_for, resolve_pitch_cache
from polyddsp.preprocess.cli import (
    PreprocessOptions,
    main,
    preprocess_options_from_cfg,
    run_preprocess,
    validate_preprocess_options,
)
from polyddsp.preprocess.audio import load_resample
from polyddsp.preprocess.ground_truth import cache_suffix as ground_truth_cache_suffix
from polyddsp.preprocess.ground_truth import midi_to_voices as ground_truth_to_voices
from polyddsp.preprocess.ground_truth import precompute_one as precompute_one_ground_truth
from polyddsp.preprocess.neutone_amt import (
    MIDI_VELOCITY_64,
    NATIVE_SAMPLE_RATE as NEUTONE_NATIVE_SR,
    audio_within_duration_limit,
)
from polyddsp.preprocess.neutone_amt import cache_suffix as neutone_cache_suffix
from polyddsp.preprocess.neutone_amt import load_model as load_neutone_amt_model
from polyddsp.preprocess.neutone_amt import precompute_one as precompute_one_neutone_amt
from polyddsp.preprocess.neutone_amt import source_for_model as neutone_amt_source
from polyddsp.preprocess.neutone_amt import to_voices as neutone_amt_to_voices
from polyddsp.preprocess.neutone_amt import validate_source as validate_neutone_amt_source

__all__ = [
    "MIDI_VELOCITY_64",
    "NEUTONE_NATIVE_SR",
    "PreprocessOptions",
    "audio_within_duration_limit",
    "bp_cache_suffix",
    "cache_path_for",
    "ground_truth_cache_suffix",
    "ground_truth_to_voices",
    "load_neutone_amt_model",
    "load_resample",
    "main",
    "neutone_amt_source",
    "neutone_amt_to_voices",
    "neutone_cache_suffix",
    "precompute_one_bp",
    "precompute_one_ground_truth",
    "precompute_one_neutone_amt",
    "preprocess_options_from_cfg",
    "resolve_pitch_cache",
    "run_preprocess",
    "validate_neutone_amt_source",
    "validate_preprocess_options",
]
