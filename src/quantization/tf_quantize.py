"""Compatibility wrapper for canonical TFLite export modules."""

from wingbeat_ml.export import *  # noqa: F401,F403
from wingbeat_ml.export.bundle import *  # noqa: F401,F403
from wingbeat_ml.export.tflite import *  # noqa: F401,F403
from wingbeat_ml.export.verify import *  # noqa: F401,F403
from wingbeat_ml.pipelines.export import (
    main,
    run_basic_quantization_suite,
)

__all__ = [
    "main",
    "run_basic_quantization_suite",
]


if __name__ == "__main__":
    main()
