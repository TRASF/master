"""Pipeline to run complete 3-stage Conv1D filter analysis on MosSong+ models."""

import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from wingbeat_ml.analysis.model.conv_filter_analysis import (
    analyze_activations,
    analyze_interventions,
    analyze_weights,
    build_representation_table,
)
from wingbeat_ml.data.loading import DataLoader
from wingbeat_ml.classification.models import MosSongPlusModel


def build_model_from_weights_h5(weights_path: str, model_config_path: str = "configs/models/mossong_plus.yaml") -> keras.Model:
    """Reconstruct exact Keras model graph from .weights.h5 or run_metadata.json."""
    path = Path(weights_path)
    meta_path = path.parent / "run_metadata.json"
    if not meta_path.exists():
        meta_path = path.parent.parent / "run_metadata.json"

    model_overrides = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_overrides = meta.get("model")
        except Exception:
            pass

    # Try building with MosSongPlusModel builder first
    import yaml
    raw_cfg = {}
    if os.path.exists(model_config_path):
        raw_cfg = yaml.safe_load(Path(model_config_path).read_text(encoding="utf-8"))

    try:
        model = MosSongPlusModel(raw_cfg, model_overrides=model_overrides).build(
            input_shape=(2400, 1),
            output_units=11,
            output_activation="softmax",
        )
        model.load_weights(weights_path)
        return model
    except Exception:
        pass

    # Fallback to inspecting h5py shapes directly for legacy/experimental checkpoints
    with h5py.File(weights_path, "r") as f:
        layers_grp = f["layers"]
        k1_shape = layers_grp["conv1d"]["vars"]["0"].shape
        k2_shape = layers_grp["conv1d_1"]["vars"]["0"].shape
        k3_shape = layers_grp["conv1d_2"]["vars"]["0"].shape
        d1_shape = layers_grp["dense"]["vars"]["0"].shape

    k1_len = k1_shape[0]
    k2_len = k2_shape[0]
    k3_len = k3_shape[0]
    c3_out = k3_shape[2]
    pool_timesteps = d1_shape[0] // c3_out
    pool_stride = max(1, 38 // max(1, pool_timesteps))

    inputs = layers.Input(shape=(2400, 1), name="input_layer")
    x = layers.Conv1D(filters=k1_shape[2], kernel_size=k1_len, strides=4, padding="same", use_bias=True, name="conv1d")(inputs)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.ReLU(name="activation")(x)

    x = layers.Conv1D(filters=k2_shape[2], kernel_size=k2_len, strides=4, padding="same", use_bias=True, name="conv1d_1")(x)
    x = layers.BatchNormalization(name="batch_normalization_1")(x)
    x = layers.ReLU(name="activation_1")(x)

    x = layers.Conv1D(filters=k3_shape[2], kernel_size=k3_len, strides=4, padding="same", use_bias=True, name="conv1d_2")(x)
    x = layers.BatchNormalization(name="batch_normalization_2")(x)
    x = layers.ReLU(name="activation_2")(x)

    x = layers.MaxPooling1D(pool_size=pool_stride, strides=pool_stride, padding="valid", name="max_pooling1d")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.Dense(d1_shape[1], use_bias=True, name="dense")(x)
    x = layers.ReLU(name="activation_3")(x)
    x = layers.Dense(128, use_bias=True, name="dense_1")(x)
    x = layers.ReLU(name="activation_4")(x)
    outputs = layers.Dense(11, activation="softmax", use_bias=True, name="dense_2")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MosquitoSongPlus")
    model.load_weights(weights_path)
    return model


def load_model_or_weights(
    model_path: str,
    model_config_path: str = "configs/models/mossong_plus.yaml",
) -> keras.Model:
    """Load a complete Keras model file or reconstruct graph and load weight file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not str(model_path).endswith(".weights.h5"):
        try:
            model = keras.models.load_model(model_path, compile=False)
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            return model
        except Exception:
            pass

    model = build_model_from_weights_h5(model_path, model_config_path)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def run_filter_analysis_pipeline(
    model_path: str,
    data_dir: str | None = None,
    output_dir: str = "output/filter_analysis",
    fs: float = 8000.0,
    initial_weights_path: str | None = None,
    model_config_path: str = "configs/models/mossong_plus.yaml",
    samples_per_class: int = 25,
) -> dict[str, pd.DataFrame]:
    """Execute complete 3-stage filter diagnostic pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1] Loading model/weights from: {model_path}")
    model = load_model_or_weights(model_path, model_config_path)

    initial_kernel = None
    if initial_weights_path and os.path.exists(initial_weights_path):
        print(f"Loading initial model weights for drift analysis: {initial_weights_path}")
        init_model = load_model_or_weights(initial_weights_path, model_config_path)
        initial_kernel = init_model.get_layer("conv1d").get_weights()[0]

    # Stage 1: Weight Analysis
    df_weights, sim_matrix, extra = analyze_weights(
        model, fs=fs, initial_kernel=initial_kernel
    )
    df_weights.to_csv(out_path / "stage1_weight_analysis.csv", index=False)
    np.save(out_path / "stage1_similarity_matrix.npy", sim_matrix)

    print("\n=== STAGE 1: WEIGHT & FREQUENCY ANALYSIS ===")
    print(df_weights.round(3).to_string(index=False))

    results = {
        "weights": df_weights,
        "similarity": pd.DataFrame(sim_matrix),
    }

    # Stage 2 & 3 if data_dir is provided
    if data_dir and os.path.exists(data_dir):
        print(f"\n[Stage 2 & 3] Loading balanced sample dataset from: {data_dir}")
        loader = DataLoader(data_dir, sample_rate=int(fs))
        file_paths, labels = loader.gather_files()

        if len(file_paths) > 0:
            samples = []
            selected_labels = []
            num_classes = loader.num_classes

            for c in range(num_classes):
                c_indices = np.where(labels == c)[0][:samples_per_class]
                for idx in c_indices:
                    audio = loader.load_file(file_paths[idx])
                    # Preprocessing matching MosSong+ pipeline: DC removal & RMS normalization
                    audio = audio - np.mean(audio)
                    rms = np.sqrt(np.mean(audio**2)) + 1e-8
                    audio = audio * (0.05 / rms)

                    if len(audio) >= 2400:
                        audio = audio[:2400]
                    else:
                        audio = np.pad(audio, (0, 2400 - len(audio)))
                    samples.append(audio[:, None])
                    selected_labels.append(c)

            x_data = np.stack(samples, axis=0).astype(np.float32)
            y_idx = np.array(selected_labels, dtype=np.int32)

            y_data = np.zeros((len(y_idx), num_classes), dtype=np.float32)
            for i, l in enumerate(y_idx):
                y_data[i, l] = 1.0

            # Diagnostics block
            base_preds = model.predict(x_data, verbose=0)
            y_true = np.argmax(y_data, axis=-1)
            y_pred = np.argmax(base_preds, axis=-1)

            print("\n--- DATASET & MODEL PREDICTION DIAGNOSTICS ---")
            print(f"x_data shape: {x_data.shape}")
            print(f"y_data shape: {y_data.shape}")
            print(f"predictions shape: {base_preds.shape}")
            print(f"True label distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")
            print(f"Predicted label distribution: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
            print(f"First 10 True labels: {y_true[:10]}")
            print(f"First 10 Pred labels: {y_pred[:10]}")
            print(f"Manual baseline accuracy: {np.mean(y_true == y_pred):.4f}")

            df_act, df_class_rms, z_saved = analyze_activations(
                model, x_data, y_data, layer_name="conv1d", class_names=loader.classes
            )
            df_act.to_csv(out_path / "stage2_activation_analysis.csv", index=False)
            if df_class_rms is not None:
                df_class_rms.to_csv(out_path / "stage2_class_rms.csv")

            print("\n=== STAGE 2: ACTIVATION ANALYSIS ===")
            print(df_act.round(3).to_string(index=False))

            df_interventions = analyze_interventions(
                model, x_data, y_data, layer_name="conv1d", class_names=loader.classes
            )
            df_interventions.to_csv(
                out_path / "stage3_interventions.csv", index=False
            )

            print("\n=== STAGE 3: INTERVENTION ANALYSIS ===")
            print(df_interventions.round(4).to_string(index=False))

            rep_table = build_representation_table(df_weights, df_act, df_interventions)
            rep_table.to_csv(out_path / "representation_summary.csv", index=False)

            print("\n=== UNIFIED REPRESENTATION SUMMARY TABLE ===")
            print(rep_table.round(4).to_string(index=False))

            results["activations"] = df_act
            results["interventions"] = df_interventions
            results["representation_table"] = rep_table

    print(f"\nAnalysis results saved to: {out_path.resolve()}")
    return results


def main():
    parser = argparse.ArgumentParser(description="MosSong+ Conv1D Filter Analysis")
    parser.add_argument("--model", type=str, required=True, help="Path to model file (.h5 / .keras / .weights.h5)")
    parser.add_argument("--model-config", type=str, default="configs/models/mossong_plus.yaml", help="Path to model config")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to audio directory")
    parser.add_argument("--output-dir", type=str, default="output/filter_analysis", help="Output directory")
    parser.add_argument("--fs", type=float, default=8000.0, help="Sampling frequency (Hz)")
    parser.add_argument("--initial-weights", type=str, default=None, help="Initial weights file")
    parser.add_argument("--samples-per-class", type=int, default=25, help="Number of balanced evaluation samples per class")
    args = parser.parse_args()

    run_filter_analysis_pipeline(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        initial_weights_path=args.initial_weights,
        model_config_path=args.model_config,
        samples_per_class=args.samples_per_class,
    )


if __name__ == "__main__":
    main()
