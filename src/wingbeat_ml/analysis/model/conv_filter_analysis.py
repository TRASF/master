"""Conv1D filter analysis suite: weights, activations, and interventions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import keras


def contiguous_3db_band(
    freq_axis: np.ndarray,
    response_db: np.ndarray,
    peak_index: int,
    threshold: float = -3.0,
) -> tuple[float, float, float]:
    """Calculate contiguous -3 dB bandwidth around peak frequency."""
    left = peak_index
    while left > 0 and response_db[left - 1] >= threshold:
        left -= 1

    right = peak_index
    while right < len(response_db) - 1 and response_db[right + 1] >= threshold:
        right += 1

    low_hz = float(freq_axis[left])
    high_hz = float(freq_axis[right])
    bw_hz = float(high_hz - low_hz)
    return low_hz, high_hz, bw_hz


def analyze_weights(
    kernel: np.ndarray | keras.Model | tf.keras.layers.Layer,
    fs: float = 8000.0,
    n_fft: int = 8192,
    initial_kernel: np.ndarray | None = None,
    bands: tuple[float, float, float] = (400.0, 1200.0, 3000.0),
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Perform Stage 1 weight & kernel-frequency analysis for first mono Conv1D layer."""
    if isinstance(kernel, keras.Model):
        kernel = kernel.get_layer("conv1d").get_weights()[0]
    elif hasattr(kernel, "get_weights"):
        kernel = kernel.get_weights()[0]

    if kernel.ndim != 3:
        raise ValueError(f"Expected [K, Cin, Cout], got {kernel.shape}")

    if kernel.shape[1] != 1:
        raise ValueError(
            "This frequency-response analysis is physically interpretable "
            "only for the mono first Conv1D. "
            f"Received Cin={kernel.shape[1]}."
        )

    kernel = kernel[:, 0, :]

    kernel_size, n_filters = kernel.shape
    low_end, primary_end, harmonic_end = bands

    H = np.fft.rfft(kernel, n=n_fft, axis=0)
    freq = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mag = np.abs(H)
    power = mag**2

    mag_db = 20.0 * np.log10(mag + 1e-12)
    mag_db_norm = mag_db - np.max(mag_db, axis=0, keepdims=True)

    peak_idx = np.argmax(mag, axis=0)
    peak_freq = freq[peak_idx]

    total_power = power.sum(axis=0) + 1e-12

    low_ratio = power[freq < low_end].sum(axis=0) / total_power
    primary_ratio = (
        power[(freq >= low_end) & (freq <= primary_end)].sum(axis=0) / total_power
    )
    harmonic_ratio = (
        power[(freq > primary_end) & (freq <= harmonic_end)].sum(axis=0)
        / total_power
    )
    upper_ratio = power[freq > harmonic_end].sum(axis=0) / total_power

    band_low = np.zeros(n_filters)
    band_high = np.zeros(n_filters)
    bandwidth = np.zeros(n_filters)

    for k in range(n_filters):
        band_low[k], band_high[k], bandwidth[k] = contiguous_3db_band(
            freq, mag_db_norm[:, k], peak_idx[k]
        )

    q_factor = np.divide(
        peak_freq,
        bandwidth,
        out=np.full_like(peak_freq, np.nan),
        where=bandwidth > 0,
    )

    spectral_centroid = (freq[:, None] * power).sum(axis=0) / total_power
    spectral_spread = np.sqrt(
        (((freq[:, None] - spectral_centroid[None, :]) ** 2) * power).sum(axis=0)
        / total_power
    )

    p = power / total_power[None, :]
    spectral_entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(freq))

    dc_db = mag_db_norm[0, :]
    nyquist_db = mag_db_norm[-1, :]

    kernel_norm = np.linalg.norm(kernel, axis=0)
    symmetry_error = (
        np.linalg.norm(kernel - kernel[::-1, :], axis=0) / (kernel_norm + 1e-12)
    )
    kernel_mean = kernel.mean(axis=0)

    # Redundancy cosine similarity matrix
    response_norm = power / (np.linalg.norm(power, axis=0, keepdims=True) + 1e-12)
    similarity = response_norm.T @ response_norm

    drift_data = {}
    if initial_kernel is not None:
        if initial_kernel.ndim == 3:
            if initial_kernel.shape[1] != 1:
                raise ValueError(
                    f"Initial kernel Cin must be 1 for mono Conv1D analysis, got {initial_kernel.shape[1]}"
                )
            initial_kernel = initial_kernel[:, 0, :]

        H_init = np.fft.rfft(initial_kernel, n=n_fft, axis=0)
        peak_idx_init = np.argmax(np.abs(H_init), axis=0)
        peak_freq_init = freq[peak_idx_init]

        rel_change = np.linalg.norm(kernel - initial_kernel, axis=0) / (
            np.linalg.norm(initial_kernel, axis=0) + 1e-12
        )
        weight_cosine = np.sum(initial_kernel * kernel, axis=0) / (
            np.linalg.norm(initial_kernel, axis=0) * kernel_norm + 1e-12
        )
        peak_freq_shift = peak_freq - peak_freq_init

        drift_data = {
            "initial_peak_hz": peak_freq_init,
            "peak_shift_hz": peak_freq_shift,
            "relative_l2_change": rel_change,
            "weight_cosine_sim": weight_cosine,
        }

    table_dict = {
        "filter": np.arange(n_filters),
        "peak_hz": peak_freq,
        "band_low_hz": band_low,
        "band_high_hz": band_high,
        "bandwidth_3db_hz": bandwidth,
        "q_factor": q_factor,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_spread_hz": spectral_spread,
        "spectral_entropy": spectral_entropy,
        "below_400_pct": 100 * low_ratio,
        "primary_band_pct": 100 * primary_ratio,
        "400_1200_pct": 100 * primary_ratio,
        "harmonic_band_pct": 100 * harmonic_ratio,
        "1200_3000_pct": 100 * harmonic_ratio,
        "above_3000_pct": 100 * upper_ratio,
        "dc_relative_db": dc_db,
        "nyquist_relative_db": nyquist_db,
        "kernel_l2_norm": kernel_norm,
        "kernel_mean": kernel_mean,
        "symmetry_error": symmetry_error,
    }
    table_dict.update(drift_data)

    df_results = pd.DataFrame(table_dict)
    extra_meta = {
        "freq": freq,
        "mag_db_norm": mag_db_norm,
        "power": power,
        "kernel": kernel,
    }
    return df_results, similarity, extra_meta


def make_conv_preactivation_probe(
    model: keras.Model,
    layer_name: str = "conv1d",
) -> keras.Model:
    """Construct a probe model that outputs signed pre-activation convolution values."""
    layer = model.get_layer(layer_name)

    if not isinstance(layer, keras.layers.Conv1D):
        raise TypeError(f"Layer {layer_name!r} must be Conv1D, got {type(layer)}")

    if layer.activation is None or getattr(layer.activation, "__name__", "") in ("linear", "<lambda>"):
        return keras.Model(
            inputs=model.input,
            outputs=layer.output,
            name=f"{layer_name}_preactivation_probe",
        )

    x = layer.input
    z = layer.convolution_op(x, layer.kernel)
    if layer.use_bias:
        z = z + layer.bias

    return keras.Model(
        inputs=model.input,
        outputs=z,
        name=f"{layer_name}_preactivation_probe",
    )


def analyze_activations(
    model: keras.Model,
    x_data: np.ndarray | tf.data.Dataset,
    y_data: np.ndarray | None = None,
    layer_name: str = "conv1d",
    batch_size: int = 128,
    class_names: list[str] | None = None,
    max_saved_samples: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame | None, np.ndarray]:
    """Perform Stage 2 activation probing using streaming accumulation and pre/post probing."""
    target_layer = model.get_layer(layer_name)
    pre_probe = make_conv_preactivation_probe(model, layer_name)

    post_probe = None
    if target_layer.activation is None or getattr(target_layer.activation, "__name__", "") in ("linear", "<lambda>"):
        for l in model.layers:
            if isinstance(l, (keras.layers.ReLU, keras.layers.Activation)) and hasattr(l, "input") and l.input is target_layer.output:
                post_probe = keras.Model(inputs=model.input, outputs=l.output, name=f"{layer_name}_postactivation_probe")
                break

    if post_probe is None:
        post_probe = keras.Model(inputs=model.input, outputs=target_layer.output, name=f"{layer_name}_postactivation_probe")

    n_filters = pre_probe.output_shape[-1]

    sum_z_pre_sq = np.zeros(n_filters, dtype=np.float64)
    sum_z_pre_neg_sq = np.zeros(n_filters, dtype=np.float64)
    count_z_pre_pos = np.zeros(n_filters, dtype=np.int64)
    count_z_pre_neg = np.zeros(n_filters, dtype=np.int64)

    sum_z_post_sq = np.zeros(n_filters, dtype=np.float64)
    count_z_post_pos = np.zeros(n_filters, dtype=np.int64)
    count_z_post_near_zero = np.zeros(n_filters, dtype=np.int64)

    total_samples = 0
    total_timesteps = 0

    class_sum_sq: dict[str, np.ndarray] = {}
    class_counts: dict[str, int] = {}

    saved_z_list = []

    if isinstance(x_data, tf.data.Dataset):
        def _ds_gen():
            for item in x_data:
                if isinstance(item, (tuple, list)):
                    yield item[0], item[1]
                else:
                    yield item, None
        iterator = _ds_gen()
    else:
        def _np_gen():
            n_samples = len(x_data)
            for i in range(0, n_samples, batch_size):
                x_b = x_data[i : i + batch_size]
                y_b = y_data[i : i + batch_size] if y_data is not None else None
                yield x_b, y_b
        iterator = _np_gen()

    for x_b, y_b in iterator:
        z_pre_b = pre_probe(x_b, training=False).numpy()
        z_post_b = post_probe(x_b, training=False).numpy()
        if y_b is not None and hasattr(y_b, "numpy"):
            y_b = y_b.numpy()

        batch_N, batch_T, _ = z_pre_b.shape
        total_samples += batch_N
        total_timesteps += batch_N * batch_T

        sum_z_pre_sq += np.sum(z_pre_b**2, axis=(0, 1))
        sum_z_pre_neg_sq += np.sum(np.where(z_pre_b < 0, z_pre_b**2, 0.0), axis=(0, 1))
        count_z_pre_pos += np.sum(z_pre_b > 0, axis=(0, 1))
        count_z_pre_neg += np.sum(z_pre_b < 0, axis=(0, 1))

        sum_z_post_sq += np.sum(z_post_b**2, axis=(0, 1))
        count_z_post_pos += np.sum(z_post_b > 0, axis=(0, 1))
        count_z_post_near_zero += np.sum(np.abs(z_post_b) < 1e-6, axis=(0, 1))

        if y_b is not None:
            if y_b.ndim > 1 and y_b.shape[1] > 1:
                y_cls_batch = np.argmax(y_b, axis=1)
            else:
                y_cls_batch = np.asarray(y_b).squeeze()

            for c_idx in np.unique(y_cls_batch):
                cls_key = (
                    class_names[c_idx]
                    if class_names and c_idx < len(class_names)
                    else f"Class_{c_idx}"
                )
                mask = y_cls_batch == c_idx
                z_cls = z_post_b[mask]
                if len(z_cls) > 0:
                    if cls_key not in class_sum_sq:
                        class_sum_sq[cls_key] = np.zeros(n_filters, dtype=np.float64)
                        class_counts[cls_key] = 0
                    class_sum_sq[cls_key] += np.sum(z_cls**2, axis=(0, 1))
                    class_counts[cls_key] += len(z_cls) * batch_T

        if total_samples - batch_N < max_saved_samples:
            take_n = min(batch_N, max_saved_samples - (total_samples - batch_N))
            saved_z_list.append(z_post_b[:take_n])

    z_saved = (
        np.concatenate(saved_z_list, axis=0)
        if saved_z_list
        else np.zeros((0, 0, n_filters), dtype=np.float32)
    )

    total_elem = float(total_timesteps) + 1e-12
    pre_rms = np.sqrt(sum_z_pre_sq / total_elem)
    post_rms = np.sqrt(sum_z_post_sq / total_elem)

    positive_fraction = count_z_pre_pos / total_elem
    negative_fraction = count_z_pre_neg / total_elem
    negative_energy_pct = 100.0 * (sum_z_pre_neg_sq / (sum_z_pre_sq + 1e-12))
    near_zero_pct = 100.0 * (count_z_post_near_zero / total_elem)

    act_df = pd.DataFrame({
        "filter": np.arange(n_filters),
        "rms": post_rms,
        "pre_activation_rms": pre_rms,
        "positive_pct": 100.0 * positive_fraction,
        "negative_pct": 100.0 * negative_fraction,
        "near_zero_pct": near_zero_pct,
        "negative_energy_pct": negative_energy_pct,
    })

    class_rms_df = None
    if class_sum_sq:
        class_rms_dict = {
            cls_key: np.sqrt(class_sum_sq[cls_key] / (float(class_counts[cls_key]) + 1e-12))
            for cls_key in class_sum_sq
        }
        class_rms_df = pd.DataFrame(
            class_rms_dict, index=[f"F{k:02d}" for k in range(n_filters)]
        ).T

    return act_df, class_rms_df, z_saved


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, float | dict[str, float]]:
    """Compute accuracy, balanced accuracy, macro-F1, and per-class recall."""
    recalls = np.zeros(n_classes, dtype=np.float64)
    precisions = np.zeros(n_classes, dtype=np.float64)
    f1s = np.zeros(n_classes, dtype=np.float64)

    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))

        recalls[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if precisions[c] + recalls[c] > 0:
            f1s[c] = 2.0 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])
        else:
            f1s[c] = 0.0

    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    class_recalls = {}
    for c in range(n_classes):
        name = class_names[c] if class_names and c < len(class_names) else f"Class_{c}"
        class_recalls[name] = float(recalls[c])

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "class_recalls": class_recalls,
    }


def analyze_interventions(
    model: keras.Model,
    x_data: np.ndarray,
    y_data: np.ndarray,
    filter_indices: list[int] | None = None,
    layer_name: str = "conv1d",
    batch_size: int = 128,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """Perform Stage 3 activation channel ablation interventions."""
    target_layer = model.get_layer(layer_name)

    try:
        feature_model = keras.Model(inputs=model.input, outputs=target_layer.output)
        tail_model = keras.Model(inputs=target_layer.output, outputs=model.output)
        use_submodels = True
    except Exception:
        use_submodels = False

    if y_data.ndim > 1 and y_data.shape[1] > 1:
        y_true = np.argmax(y_data, axis=1)
        n_classes = y_data.shape[1]
    else:
        y_true = np.asarray(y_data).squeeze()
        n_classes = int(np.max(y_true)) + 1 if len(y_true) > 0 else 11

    if use_submodels:
        z_base = feature_model.predict(x_data, batch_size=batch_size, verbose=0)
        base_preds = tail_model.predict(z_base, batch_size=batch_size, verbose=0)
    else:
        base_preds = model.predict(x_data, batch_size=batch_size, verbose=0)
        z_base = None

    n_filters = (
        z_base.shape[-1]
        if z_base is not None
        else target_layer.get_weights()[0].shape[-1]
    )

    if filter_indices is None:
        filter_indices = list(range(n_filters))

    y_base_pred = np.argmax(base_preds, axis=1)
    base_metrics = compute_classification_metrics(y_true, y_base_pred, n_classes, class_names)

    results = []

    if use_submodels:
        for f_idx in filter_indices:
            z_ablated = z_base.copy()
            z_ablated[:, :, f_idx] = 0.0
            ablated_preds = tail_model.predict(
                z_ablated, batch_size=batch_size, verbose=0
            )
            y_ablated_pred = np.argmax(ablated_preds, axis=1)

            ablated_metrics = compute_classification_metrics(
                y_true, y_ablated_pred, n_classes, class_names
            )

            acc_drop = base_metrics["accuracy"] - ablated_metrics["accuracy"]
            bal_acc_drop = (
                base_metrics["balanced_accuracy"]
                - ablated_metrics["balanced_accuracy"]
            )
            macro_f1_drop = base_metrics["macro_f1"] - ablated_metrics["macro_f1"]
            disagreement_rate = float(np.mean(y_base_pred != y_ablated_pred))
            mean_prob_change = float(np.mean(np.abs(base_preds - ablated_preds)))

            res_row = {
                "filter": f_idx,
                "baseline_accuracy": base_metrics["accuracy"],
                "ablated_accuracy": ablated_metrics["accuracy"],
                "accuracy_drop": acc_drop,
                "baseline_balanced_accuracy": base_metrics["balanced_accuracy"],
                "ablated_balanced_accuracy": ablated_metrics["balanced_accuracy"],
                "balanced_accuracy_drop": bal_acc_drop,
                "baseline_macro_f1": base_metrics["macro_f1"],
                "ablated_macro_f1": ablated_metrics["macro_f1"],
                "macro_f1_drop": macro_f1_drop,
                "disagreement_rate": disagreement_rate,
                "mean_prob_change": mean_prob_change,
            }

            for cls_name, b_rec in base_metrics["class_recalls"].items():
                a_rec = ablated_metrics["class_recalls"][cls_name]
                res_row[f"{cls_name}_recall_drop"] = b_rec - a_rec

            results.append(res_row)
    else:
        # Fallback to safe weight zeroing inside try/finally if submodel graph split fails
        weights_orig = target_layer.get_weights()
        kernel_orig = weights_orig[0].copy()
        try:
            for f_idx in filter_indices:
                kernel_ablated = kernel_orig.copy()
                kernel_ablated[:, :, f_idx] = 0.0
                target_layer.set_weights([kernel_ablated] + weights_orig[1:])

                ablated_preds = model.predict(x_data, batch_size=batch_size, verbose=0)
                y_ablated_pred = np.argmax(ablated_preds, axis=1)

                ablated_metrics = compute_classification_metrics(
                    y_true, y_ablated_pred, n_classes, class_names
                )

                acc_drop = base_metrics["accuracy"] - ablated_metrics["accuracy"]
                bal_acc_drop = (
                    base_metrics["balanced_accuracy"]
                    - ablated_metrics["balanced_accuracy"]
                )
                macro_f1_drop = base_metrics["macro_f1"] - ablated_metrics["macro_f1"]
                disagreement_rate = float(np.mean(y_base_pred != y_ablated_pred))
                mean_prob_change = float(np.mean(np.abs(base_preds - ablated_preds)))

                res_row = {
                    "filter": f_idx,
                    "baseline_accuracy": base_metrics["accuracy"],
                    "ablated_accuracy": ablated_metrics["accuracy"],
                    "accuracy_drop": acc_drop,
                    "baseline_balanced_accuracy": base_metrics["balanced_accuracy"],
                    "ablated_balanced_accuracy": ablated_metrics["balanced_accuracy"],
                    "balanced_accuracy_drop": bal_acc_drop,
                    "baseline_macro_f1": base_metrics["macro_f1"],
                    "ablated_macro_f1": ablated_metrics["macro_f1"],
                    "macro_f1_drop": macro_f1_drop,
                    "disagreement_rate": disagreement_rate,
                    "mean_prob_change": mean_prob_change,
                }

                for cls_name, b_rec in base_metrics["class_recalls"].items():
                    a_rec = ablated_metrics["class_recalls"][cls_name]
                    res_row[f"{cls_name}_recall_drop"] = b_rec - a_rec

                results.append(res_row)
        finally:
            target_layer.set_weights(weights_orig)

    return pd.DataFrame(results)


def build_representation_table(
    df_weights: pd.DataFrame,
    df_activations: pd.DataFrame,
    df_interventions: pd.DataFrame,
    activation_corr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build unified representation summary table merging weight, activation, and intervention metrics."""
    merged = df_weights.copy()

    act_cols = [c for c in df_activations.columns if c != "filter" and c not in merged.columns]
    merged = merged.merge(df_activations[["filter"] + act_cols], on="filter", how="left")

    inv_cols = [c for c in df_interventions.columns if c != "filter" and c not in merged.columns]
    merged = merged.merge(df_interventions[["filter"] + inv_cols], on="filter", how="left")

    if activation_corr is not None and activation_corr.shape == (len(merged), len(merged)):
        corr_matrix = np.abs(activation_corr.copy())
        np.fill_diagonal(corr_matrix, 0.0)
        max_redundancy = np.max(corr_matrix, axis=1)
        merged["activation_redundancy"] = max_redundancy

    return merged


__all__ = [
    "analyze_activations",
    "analyze_interventions",
    "analyze_weights",
    "build_representation_table",
    "contiguous_3db_band",
    "make_conv_preactivation_probe",
]


if __name__ == "__main__":
    import os
    from pathlib import Path
    import yaml

    from wingbeat_ml.data.loading import DataLoader
    from wingbeat_ml.classification.models import MosSongPlusModel

    config_path = Path("configs/models/mossong_plus.yaml")
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        raise SystemExit(1)

    print(f"Loading MosSong+ configuration: {config_path}")
    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = MosSongPlusModel(raw_cfg).build(
        input_shape=(2400, 1), output_units=11, output_activation="softmax"
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    print("\n=== STAGE 1: WEIGHT & FREQUENCY ANALYSIS ===")
    conv1_kernel = model.get_layer("conv1d").get_weights()[0]
    df_weights, sim_matrix, _ = analyze_weights(conv1_kernel, fs=8000.0)
    print(
        df_weights[
            [
                "filter",
                "peak_hz",
                "bandwidth_3db_hz",
                "q_factor",
                "spectral_entropy",
                "symmetry_error",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    data_dirs = ["dataset/MSB/Indoor", "dataset/temp_indoor_val"]
    chosen_dir = next((d for d in data_dirs if os.path.exists(d)), None)

    if chosen_dir:
        print(f"\n=== LOADING SAMPLE DATA FROM: {chosen_dir} ===")
        loader = DataLoader(chosen_dir, sample_rate=8000)
        file_paths, labels = loader.gather_files()
        samples = []
        oh_labels = []
        for p, l in zip(file_paths[:32], labels[:32]):
            audio = loader.load_file(p)
            if len(audio) >= 2400:
                audio = audio[:2400]
            else:
                audio = np.pad(audio, (0, 2400 - len(audio)))
            samples.append(audio[:, None])
            oh = np.zeros(11, dtype=np.float32)
            oh[l] = 1.0
            oh_labels.append(oh)

        x_data = np.stack(samples, axis=0).astype(np.float32)
        y_data = np.stack(oh_labels, axis=0).astype(np.float32)

        print("\n=== STAGE 2: ACTIVATION PROBING ===")
        df_act, df_class_rms, z_saved = analyze_activations(
            model, x_data, y_data, layer_name="conv1d", class_names=loader.classes
        )
        print(df_act.head(10).round(3).to_string(index=False))

        print("\n=== STAGE 3: ACTIVATION CHANNEL ABLATION INTERVENTIONS ===")
        df_inv = analyze_interventions(
            model,
            x_data,
            y_data,
            filter_indices=list(range(5)),
            layer_name="conv1d",
            class_names=loader.classes,
        )
        print(
            df_inv[
                [
                    "filter",
                    "baseline_accuracy",
                    "ablated_accuracy",
                    "accuracy_drop",
                    "macro_f1_drop",
                    "disagreement_rate",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

        print("\n=== UNIFIED REPRESENTATION SUMMARY TABLE ===")
        rep_table = build_representation_table(df_weights, df_act, df_inv)
        print(
            rep_table[
                [
                    "filter",
                    "peak_hz",
                    "bandwidth_3db_hz",
                    "rms",
                    "pre_activation_rms",
                    "negative_energy_pct",
                    "accuracy_drop",
                    "macro_f1_drop",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

