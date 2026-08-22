"""Pipeline entry point for SignalGrad-CAM post-hoc XAI analysis using real dataset samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from wingbeat_ml.config import load_config
from wingbeat_ml.config.schema import AppConfig
from wingbeat_ml.analysis.model.signal_gradcam import (
    CamDiagnostics,
    ExplanationSample,
    GradCamConfig,
    SignalGradCamAnalyzer,
    collect_real_samples_by_class,
)
from wingbeat_ml.classification.pipelines.helpers.components import build_dataset_bundle
from wingbeat_ml.visualizer.signal_gradcam import plot_signal_gradcam
from wingbeat_ml.pipelines.signal_gradcam_runtime import explain_sample

CONV_LAYER_CLASS_KEYWORDS = (
    "conv1d",
    "sincconv1d",
    "repconv1d",
    "separableconv1d",
    "depthwiseconv1d",
)


def load_exact_model_and_config(
    checkpoint_path: str,
    model_config_path: Optional[str] = None,
    defaults_path: str = "configs/defaults.yaml",
) -> Tuple[Any, AppConfig]:
    """Load exact trained model and configuration without mutating architecture settings."""
    if tf is None:
        raise ImportError("TensorFlow required for explain pipeline.")

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    exp_file = path.parent / "run_metadata.json"
    exp_path = str(exp_file) if exp_file.exists() else None

    model_path = model_config_path
    if not model_path or not Path(model_path).exists():
        if Path("configs/models/mossong_plus.yaml").exists():
            model_path = "configs/models/mossong_plus.yaml"
        elif Path("configs/best.yaml").exists():
            model_path = "configs/best.yaml"

    tf.keras.backend.clear_session()
    app_cfg = load_config(
        model_path=model_path,
        experiment_path=exp_path,
        defaults_path=defaults_path,
    )

    try:
        model = tf.keras.models.load_model(str(path))
    except Exception:
        arch_cfg = getattr(app_cfg.model, app_cfg.model.id, None)
        if arch_cfg is None:
            arch_cfg = app_cfg.model.model_dump().get(app_cfg.model.id, {})
        arch_dict = (
            arch_cfg
            if isinstance(arch_cfg, dict)
            else (arch_cfg.model_dump() if hasattr(arch_cfg, "model_dump") else dict(arch_cfg))
        )

        try:
            from wingbeat_ml.classification.pipelines.helpers.components import build_model_component

            model = build_model_component(app_cfg, arch_dict)
            model.load_weights(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Could not reconstruct the exact trained architecture from checkpoint '{checkpoint_path}'; "
                "refusing to run XAI on a modified model."
            ) from exc

    # Verification checks
    n_classes = len(app_cfg.classes)
    if model.output_shape[-1] != n_classes:
        raise RuntimeError(
            f"Model output dimension ({model.output_shape[-1]}) does not match class count ({n_classes})."
        )

    dummy_input = np.zeros((1, app_cfg.audio.segment_length, 1), dtype=np.float32)
    test_preds = np.asarray(model.predict(dummy_input, verbose=0))
    if not np.all(np.isfinite(test_preds)):
        raise RuntimeError("Model predictions contain NaN or Inf values.")

    return model, app_cfg


def discover_and_verify_conv_layers(
    model: Any, requested_layer: Optional[str] = None
) -> List[str]:
    """Dynamically discover Conv1D-like layers and verify temporal dimension retention."""
    valid_conv_layers = []
    print("\n==========================================================================================")
    print("SEARCHING FOR NETWORK CONVOLUTIONAL FEATURE LAYERS:")
    print("==========================================================================================")
    for layer in model.layers:
        class_name = layer.__class__.__name__
        class_lower = class_name.lower()
        name_lower = layer.name.lower()

        if any(keyword in class_lower or keyword in name_lower for keyword in CONV_LAYER_CLASS_KEYWORDS):
            out_shape = getattr(layer, "output_shape", None)
            if out_shape is None and hasattr(layer, "output"):
                out_shape = getattr(layer.output, "shape", None)
            print(f" - {layer.name:<25} | type: {class_name:<20} | shape: {out_shape}")
            valid_conv_layers.append(layer.name)

    if not valid_conv_layers:
        raise RuntimeError("No convolutional feature layers found in model.")

    if requested_layer and requested_layer.lower() != "all":
        found_names = [l.name for l in model.layers]
        if requested_layer not in found_names:
            raise ValueError(f"Requested layer '{requested_layer}' not found in model layers: {found_names}")
        target_obj = model.get_layer(requested_layer)
        out_shape = getattr(target_obj, "output_shape", None) or getattr(getattr(target_obj, "output", None), "shape", None)
        if len(out_shape) < 3 or (out_shape[1] is not None and out_shape[1] <= 1):
            raise ValueError(
                f"Target layer '{requested_layer}' does not retain a temporal dimension (shape: {out_shape})."
            )
        return [requested_layer]

    return valid_conv_layers


def run_explain_pipeline(
    checkpoint_path: str,
    model_config_path: Optional[str] = None,
    defaults_path: str = "configs/defaults.yaml",
    split: str = "test",
    dataset_domain: str = "all",
    samples_per_class: int = 5,
    correct_only: bool = True,
    incorrect_only: bool = False,
    seed: int = 42,
    class_filter: Optional[str] = None,
    target_layer: Optional[str] = None,
    explainer: str = "Grad-CAM",
    output_dir: str = "artifacts/explanations",
    wandb_log: bool = False,
    allow_degenerate_cam: bool = False,
) -> Dict[str, Any]:
    """Execute post-hoc SignalGrad-CAM explainability pipeline using real dataset samples."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load exact trained model
    model, app_cfg = load_exact_model_and_config(
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path,
        defaults_path=defaults_path,
    )

    # 2. Discover Conv layers
    conv_layers = discover_and_verify_conv_layers(model, requested_layer=target_layer)

    # 3. Build real evaluation dataset
    builder, train_ds, val_ds, test_ds = build_dataset_bundle(app_cfg, return_builder=True)
    if split == "test":
        eval_ds = test_ds
    elif split == "val" or split == "validation":
        eval_ds = val_ds
    elif split == "train":
        eval_ds = train_ds
    else:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train', 'val', or 'test'.")

    # 4. Collect real samples by class
    collected_by_class = collect_real_samples_by_class(
        model=model,
        dataset=eval_ds,
        class_names=app_cfg.classes,
        samples_per_class=samples_per_class,
        correct_only=correct_only,
        incorrect_only=incorrect_only,
        class_filter=class_filter,
        domain_filter=dataset_domain,
    )

    all_samples: List[ExplanationSample] = []
    print("\nReal sample collection summary:")
    for cid, samples in collected_by_class.items():
        cname = app_cfg.classes[cid]
        print(f" - Class {cid:2d} ({cname:<22}): {len(samples)} real samples collected")
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("No real dataset samples matched the requested collection criteria.")

    # 5. Sanity experiment on 1 real mosquito sample
    mosquito_samples = [s for s in all_samples if s.correct and s.true_name.casefold() != "no.mos"]
    sanity_sample = mosquito_samples[0] if mosquito_samples else all_samples[0]
    deepest_layer = conv_layers[-1]

    print("\n==========================================================================================")
    print("INITIAL SANITY EXPERIMENT ON REAL SAMPLE:")
    print("==========================================================================================")
    sig_flat = np.squeeze(sanity_sample.signal)
    rms_val = float(np.sqrt(np.mean(np.square(sig_flat))))
    print(f"Source sample ID:        {sanity_sample.sample_id}")
    print(f"True class:              {sanity_sample.true_name} (ID {sanity_sample.true_id})")
    print(f"Predicted class:         {sanity_sample.predicted_name} (ID {sanity_sample.predicted_id})")
    print(f"Confidence:              {sanity_sample.confidence:.4f}")
    print(f"Waveform Min/Max/RMS:    {np.min(sig_flat):.4f} / {np.max(sig_flat):.4f} / {rms_val:.4f}")

    sanity_exp, sanity_manifest, sanity_diag = explain_sample(
        model=model,
        class_names=app_cfg.classes,
        signal=sanity_sample.signal,
        true_class_id=sanity_sample.true_id,
        target_layer=deepest_layer,
        explainer=explainer,
        output_plot=str(out_path / "sanity_check.png"),
        sampling_rate=app_cfg.audio.sample_rate,
        target_class_id=None,
    )

    print(f"Target Conv Layer:       {deepest_layer}")
    print(f"CAM Key Matched:         {sanity_manifest.get('selected_cam_key')}")
    print(f"CAM Shape:               {sanity_diag.get('shape')}")
    print(f"CAM Min / Max / Mean:    {sanity_diag.get('cam_min', 0.0):.6f} / {sanity_diag.get('cam_max', 0.0):.6f} / {sanity_diag.get('cam_mean', 0.0):.6f}")
    print(f"CAM Nonzero Percentage:  {sanity_diag.get('nonzero_fraction', 0.0) * 100:.2f}%")
    print(f"Degenerate CAM:          {sanity_diag.get('degenerate_cam', False)}")

    if sanity_diag.get("degenerate_cam", False) and not allow_degenerate_cam:
        raise RuntimeError(
            "Initial sanity check produced a degenerate (zero/flat) CAM on real mosquito sample. "
            "Pass --allow-degenerate-cam if you explicitly wish to continue batch processing."
        )

    # 6. Process batch explanations across discovered layers
    manifest_rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    generated_plots: List[str] = []

    degenerate_count = 0
    total_explanations = 0

    print(f"\nProcessing SignalGrad-CAM explanations across {len(conv_layers)} layer(s)...")

    for layer in conv_layers:
        for sample in all_samples:
            sample_domain = sample.dataset_domain
            sub_dir = (
                out_path
                / sample_domain
                / sample.true_name
                / ("correct" if sample.correct else "incorrect")
                / sample.sample_id
            )
            sub_dir.mkdir(parents=True, exist_ok=True)
            plot_path = str(sub_dir / f"{layer}.png")

            explanation, manifest_row, diagnostic_row = explain_sample(
                model=model,
                class_names=app_cfg.classes,
                signal=sample.signal,
                true_class_id=sample.true_id,
                target_layer=layer,
                explainer=explainer,
                output_plot=plot_path,
                sampling_rate=app_cfg.audio.sample_rate,
                target_class_id=None,
            )

            manifest_row.update({
                "sample_id": sample.sample_id,
                "source_file": sample.source_file,
                "original_recording_id": sample.original_recording_id,
                "window_start_seconds": sample.window_start_seconds,
                "window_end_seconds": sample.window_end_seconds,
                "dataset_domain": sample.dataset_domain,
                "split": split,
            })

            total_explanations += 1
            if diagnostic_row.get("degenerate_cam", False):
                degenerate_count += 1

            generated_plots.append(plot_path)
            manifest_rows.append(manifest_row)
            diag_rows.append(diagnostic_row)

    # Save manifest.csv & cam_diagnostics.csv
    manifest_path = out_path / "explanation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    diag_path = out_path / "cam_diagnostics.csv"
    with diag_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diag_rows)

    deg_fraction = float(degenerate_count / max(1, total_explanations))
    print(f"\nSaved explanation manifest to: {manifest_path}")
    print(f"Saved CAM diagnostics to:       {diag_path}")
    print(f"Total explanations generated:    {total_explanations}")
    print(f"Degenerate CAM fraction:        {deg_fraction * 100:.2f}% ({degenerate_count}/{total_explanations})")

    # W&B Logging
    if wandb_log:
        try:
            import wandb

            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=f"xai_explanations_{split}",
                    type="xai_analysis",
                    description="SignalGrad-CAM post-hoc explanations on real dataset split",
                )
                artifact.add_file(str(manifest_path))
                artifact.add_file(str(diag_path))
                for plot in generated_plots[:50]:
                    artifact.add_file(plot)

                wandb.log_artifact(artifact)
                wandb.log({
                    "xai/n_samples": len(all_samples),
                    "xai/n_degenerate_cam": degenerate_count,
                    "xai/degenerate_fraction": deg_fraction,
                })
        except ImportError:
            pass

    return {
        "status": "success",
        "output_dir": str(out_path),
        "total_explanations": total_explanations,
        "degenerate_count": degenerate_count,
        "degenerate_fraction": deg_fraction,
        "manifest_path": str(manifest_path),
        "diagnostics_path": str(diag_path),
    }


def main():
    parser = argparse.ArgumentParser(description="SignalGrad-CAM Explain Pipeline with Real Dataset Samples")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .keras or .h5 checkpoint / weights")
    parser.add_argument("--model-config", help="Optional model YAML config file")
    parser.add_argument("--defaults-path", default="configs/defaults.yaml", help="Path to defaults configuration YAML")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--dataset-domain", default="all", choices=["indoor", "outdoor", "all"], help="Domain filter")
    parser.add_argument("--samples-per-class", type=int, default=5, help="Number of real samples per class")
    parser.add_argument("--correct-only", action="store_true", default=True, help="Select correctly classified samples")
    parser.add_argument("--incorrect-only", action="store_true", default=False, help="Select misclassified samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--class", dest="class_filter", help="Optional single target class name filter")
    parser.add_argument("--layer", default="all", help="Target conv layer name (omitted or 'all' scans all conv layers)")
    parser.add_argument("--method", default="Grad-CAM", choices=["Grad-CAM", "HiResCAM"], help="CAM explainer method")
    parser.add_argument("--output-dir", default="artifacts/explanations", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Log explanation artifacts to W&B")
    parser.add_argument("--allow-degenerate-cam", action="store_true", help="Override error on degenerate sanity check")

    args = parser.parse_args()

    run_explain_pipeline(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        defaults_path=args.defaults_path,
        split=args.split,
        dataset_domain=args.dataset_domain,
        samples_per_class=args.samples_per_class,
        correct_only=args.correct_only and not args.incorrect_only,
        incorrect_only=args.incorrect_only,
        seed=args.seed,
        class_filter=args.class_filter,
        target_layer=args.layer,
        explainer=args.method,
        output_dir=args.output_dir,
        wandb_log=args.wandb,
        allow_degenerate_cam=args.allow_degenerate_cam,
    )


if __name__ == "__main__":
    main()
