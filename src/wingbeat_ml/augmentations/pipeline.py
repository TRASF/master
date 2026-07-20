"""Augmentation pipeline: ordered transform selection and execution.

Provides an explicit transform registry and a factory function.

TRANSFORMS maps config key names to bound transform method names on
AudioAugmentor.  This is the canonical mapping used to verify that all
configured transform names are known at pipeline construction time.

Registry:
  "high_pass"      → augmentor.apply_hpf
  "pre_emphasis"   → augmentor.pre_emphasis
  "pitch_shift"    → augmentor.pitch_shift
  "time_shift"     → augmentor.time_shift
  "time_masking"   → augmentor.apply_time_masking
  "random_gain"    → augmentor.random_gain
  "gaussian_noise" → augmentor.add_gaussian_noise
  "noise_overlay"  → augmentor.add_noise
  mixup is applied at batch level by SupervisedDataset.

Transform application order is fixed inside AudioAugmentor.apply_post_processing
(see transforms.py docstring).  The TRANSFORMS dict here documents the mapping;
runtime ordering is owned by apply_post_processing.

No decorators, automatic discovery, or plugin loader.
"""

from __future__ import annotations

from typing import Mapping

from wingbeat_ml.augmentations.transforms import AudioAugmentor


# ---------------------------------------------------------------------------
# Explicit transform registry (config key → method name on AudioAugmentor)
# ---------------------------------------------------------------------------

TRANSFORMS: dict[str, str] = {
    "high_pass": "apply_hpf",
    "pre_emphasis": "pre_emphasis",
    "pitch_shift": "pitch_shift",
    "time_shift": "time_shift",
    "time_masking": "apply_time_masking",
    "random_gain": "random_gain",
    "gaussian_noise": "add_gaussian_noise",
    "noise_overlay": "add_noise",
    # mixup is applied at batch level; listed here for completeness
}


def build_augmentor(
    augment_cfg: Mapping[str, object],
    segment_length: int = 2400,
    seed: int = 42,
    deterministic: bool = False,
    nomos_index: int | None = None,
) -> AudioAugmentor:
    """Construct an AudioAugmentor from a configuration mapping.

    Validates that all keys in *augment_cfg* are known transform names
    or recognised top-level keys.  Raises ValueError for unknown names.

    Args:
        augment_cfg: Augmentation section of the resolved configuration.
        segment_length: Waveform segment length in samples.
        seed: Global random seed.
        deterministic: If True, pipeline uses seeded deterministic RNG.
        nomos_index: Integer index of the No.Mos (background) class.

    Returns:
        Configured AudioAugmentor instance.
    """
    # Keys that are not transform names
    _non_transform_keys = frozenset({
        "config",
        "mixup",
        "noise_banks",
        "overlap",
        "preprocess",
        "rms_norm",
    })

    unknown = []
    for key in augment_cfg:
        if key not in TRANSFORMS and key not in _non_transform_keys:
            unknown.append(key)
    if unknown:
        raise ValueError(
            f"Unknown augmentation transform name(s): {unknown}. "
            f"Recognised names: {sorted(TRANSFORMS)}"
        )

    return AudioAugmentor(
        segment_length=segment_length,
        config=dict(augment_cfg),
        seed=seed,
        deterministic=deterministic,
        nomos_index=nomos_index,
    )
