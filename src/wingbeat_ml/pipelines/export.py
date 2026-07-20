"""Reproducible TFLite export and verification pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from wingbeat_ml.export.bundle import (
    export_input_quantization_header,
    export_tflite_to_c_header,
    write_esp32_readme,
)
from wingbeat_ml.export.tflite import (
    convert_dynamic_range_tflite,
    convert_float_tflite,
    convert_full_int8_tflite,
    dump_tflite_analyzer,
    ensure_dir,
    inspect_tflite_io,
    make_representative_dataset,
    run_quantization_debugger,
)
from wingbeat_ml.export.verify import (
    compare_model_pair_agreement,
    evaluate_keras_input_qdq_model,
    evaluate_keras_model,
    evaluate_tflite_model,
)


def _write_summary(rows, path):
    fields = [
        "model",
        "size_bytes",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top1_agreement_with_keras",
        "top3_overlap_with_keras",
    ]

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_basic_quantization_suite(
    keras_model,
    val_ds,
    test_ds,
    out_dir,
    class_names=None,
    rep_samples=500,
    seed=42,
    input_amplitude_range=0.03,
    allow_dummy_calibration=False,
    run_debugger=False,
):
    """Convert, verify and bundle float/dynamic/int8 models."""
    out_dir = ensure_dir(out_dir)

    if val_ds is not None:
        representative_dataset = make_representative_dataset(
            val_ds,
            max_samples=rep_samples,
            seed=seed,
        )
    elif allow_dummy_calibration:
        def representative_dataset():
            generator = np.random.default_rng(seed)
            for _ in range(rep_samples):
                sample = generator.normal(
                    0.0,
                    0.05,
                    (1, 2400, 1),
                ).astype(np.float32)
                yield [np.clip(sample, -1.0, 1.0)]
    else:
        raise RuntimeError(
            "Full INT8 deployment requires a real representative "
            "dataset. Dummy calibration is only for smoke tests."
        )

    paths = {
        "float": convert_float_tflite(
            keras_model,
            out_dir / "model_float32.tflite",
        ),
        "dynamic": convert_dynamic_range_tflite(
            keras_model,
            out_dir / "model_dynamic_range.tflite",
        ),
        "int8": convert_full_int8_tflite(
            keras_model,
            representative_dataset,
            out_dir / "model_full_int8.tflite",
        ),
    }

    paths["int8_header"] = export_tflite_to_c_header(
        paths["int8"],
        out_dir / "model_full_int8.h",
        array_name="g_model",
    )
    paths["input_quantization"] = (
        export_input_quantization_header(
            paths["int8"],
            out_dir / "model_input_quantization.h",
            amplitude_range=input_amplitude_range,
        )
    )
    paths["readme"] = write_esp32_readme(
        out_dir,
        array_name="g_model",
    )

    results = {}
    summary_rows = []

    if test_ds is not None:
        results["keras_float"] = evaluate_keras_model(
            keras_model,
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "keras_float",
        )
        results["tflite_float"] = evaluate_tflite_model(
            paths["float"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "tflite_float",
        )
        results["tflite_dynamic"] = evaluate_tflite_model(
            paths["dynamic"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "tflite_dynamic",
        )
        results["keras_input_qdq"] = (
            evaluate_keras_input_qdq_model(
                keras_model,
                paths["int8"],
                test_ds,
                class_names=class_names,
                out_prefix=out_dir / "keras_input_qdq",
            )
        )
        results["tflite_int8"] = evaluate_tflite_model(
            paths["int8"],
            test_ds,
            class_names=class_names,
            out_prefix=out_dir / "tflite_int8",
        )

        for name, result in results.items():
            row = {
                "model": name,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            }

            path_key = {
                "tflite_float": "float",
                "tflite_dynamic": "dynamic",
                "tflite_int8": "int8",
            }.get(name)
            row["size_bytes"] = (
                paths[path_key].stat().st_size
                if path_key else ""
            )

            if name != "keras_float":
                agreement = compare_model_pair_agreement(
                    results["keras_float"]["scores"],
                    result["scores"],
                )
                row.update({
                    "top1_agreement_with_keras":
                        agreement["top1_agreement"],
                    "top3_overlap_with_keras":
                        agreement["top3_overlap"],
                })

            summary_rows.append(row)

    summary_path = out_dir / "quantization_summary.csv"
    _write_summary(summary_rows, summary_path)

    inspect_tflite_io(paths["int8"])
    dump_tflite_analyzer(
        paths["int8"],
        out_dir / "model_full_int8_analyzer.txt",
    )

    if run_debugger and val_ds is not None:
        run_quantization_debugger(
            keras_model,
            representative_dataset,
            out_dir / "quantization_debugger.csv",
        )

    return {
        "paths": paths,
        "results": results,
        "summary": summary_rows,
        "summary_path": summary_path,
    }


def export_from_weights(
    *,
    defaults_path,
    model_config_path,
    weights_path,
    out_dir,
    rep_samples=500,
    input_amplitude_range=None,
    allow_dummy_calibration=False,
    run_debugger=False,
):
    """Build MosSongPlus, load weights and run export."""
    from wingbeat_ml.config.runtime import (
        configure_training_runtime,
        load_config,
        normalize_config,
    )
    from wingbeat_ml.data.dataset import build_datasets
    from wingbeat_ml.registry import build_model

    cfg = normalize_config(load_config(defaults_path))
    model_cfg = load_config(model_config_path)

    configure_training_runtime(cfg["reproducibility"])
    seed = cfg["reproducibility"]["seed"]

    try:
        dataset_config = copy.deepcopy(cfg)
        dataset_config["train"]["shuffle"] = False
        _, val_ds, test_ds = build_datasets(
            cfg["dataset"].get("train_dir")
            or cfg["dataset"]["indoor"],
            dataset_config,
            val_dir=cfg["dataset"]["val_dir"],
            test_dir=cfg["dataset"]["test_dir"],
        )
    except Exception:
        if not allow_dummy_calibration:
            raise
        val_ds = test_ds = None

    model = build_model(
        cfg,
        model_cfg,
        batch_size=1,
    )

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}"
        )
    model.load_weights(weights_path)

    quantization = cfg.get("quantization", {})
    amplitude = (
        input_amplitude_range
        if input_amplitude_range is not None
        else quantization.get("input_amplitude_range", 0.03)
    )

    return run_basic_quantization_suite(
        keras_model=model,
        val_ds=val_ds,
        test_ds=test_ds,
        out_dir=out_dir,
        class_names=cfg["classes"],
        rep_samples=rep_samples,
        seed=seed,
        input_amplitude_range=amplitude,
        allow_dummy_calibration=allow_dummy_calibration,
        run_debugger=run_debugger,
    )


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Export MosSongPlus to TFLite/TFLite Micro",
    )
    parser.add_argument(
        "--defaults-path",
        default="configs/defaults.yaml",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--out-dir",
        default="quantization_output",
    )
    parser.add_argument("--rep-samples", type=int, default=500)
    parser.add_argument("--input-amplitude-range", type=float)
    parser.add_argument(
        "--allow-dummy-calibration",
        action="store_true",
    )
    parser.add_argument("--run-debugger", action="store_true")
    parsed = parser.parse_args(args)

    return export_from_weights(
        defaults_path=parsed.defaults_path,
        model_config_path=parsed.model_config,
        weights_path=parsed.weights,
        out_dir=parsed.out_dir,
        rep_samples=parsed.rep_samples,
        input_amplitude_range=parsed.input_amplitude_range,
        allow_dummy_calibration=parsed.allow_dummy_calibration,
        run_debugger=parsed.run_debugger,
    )


__all__ = [
    "export_from_weights",
    "main",
    "run_basic_quantization_suite",
]
