"""Public augmentation API for wingbeat_ml.

The core implementation lives in AudioAugmentor (transforms.py) and
the pipeline entry point (pipeline.py).
"""

from wingbeat_ml.augmentations.pipeline import (
    build_augmentor,
    TRANSFORMS,
)
from wingbeat_ml.augmentations.transforms import AudioAugmentor

__all__ = [
    "AudioAugmentor",
    "build_augmentor",
    "TRANSFORMS",
]
