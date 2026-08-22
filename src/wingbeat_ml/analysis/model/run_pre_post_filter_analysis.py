"""Pre- and Post-Training Conv1D Filter Analysis Runner."""

from pathlib import Path
import numpy as np
import pandas as pd
import keras

from wingbeat_ml.analysis.model.conv_filter_analysis import (
    analyze_activations,
    analyze_interventions,
    analyze_weights,
    build_representation_table,
)


def run_full_pre_post_analysis(
    model: keras.Model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_fn=None,
    layer_name: str = "conv1d",
    fs: float = 8000.0,
    output_dir: str = "output/filter_analysis_pre_post",
    class_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Automate pre-training analysis, training, post-training analysis, and drift reporting."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target_layer = model.get_layer(layer_name)
    initial_kernel = target_layer.get_weights()[0].copy()

    print("==================================================")
    print("STAGE 1: PRE-TRAINING FILTER ANALYSIS")
    print("==================================================")
    df_pre_weights, sim_pre, _ = analyze_weights(
        initial_kernel, fs=fs
    )
    df_pre_act, class_rms_pre, _ = analyze_activations(
        model, x_val, y_val, layer_name=layer_name, class_names=class_names
    )

    df_pre_weights.to_csv(out_path / "pre_training_weights.csv", index=False)
    df_pre_act.to_csv(out_path / "pre_training_activations.csv", index=False)
    np.save(out_path / "pre_training_similarity.npy", sim_pre)
    if class_rms_pre is not None:
        class_rms_pre.to_csv(out_path / "pre_training_class_rms.csv")

    print("\nPre-training weight summary (first 5 filters):")
    print(df_pre_weights[["filter", "peak_hz", "bandwidth_3db_hz", "q_factor", "spectral_entropy"]].head().round(3).to_string(index=False))

    # Optional training step
    if train_fn is not None:
        print("\nExecuting training step...")
        train_fn(model)

    trained_kernel = target_layer.get_weights()[0].copy()

    print("\n==================================================")
    print("STAGE 2: POST-TRAINING & DRIFT ANALYSIS")
    print("==================================================")
    df_post_weights, sim_post, _ = analyze_weights(
        trained_kernel, fs=fs, initial_kernel=initial_kernel
    )
    df_post_act, class_rms_post, _ = analyze_activations(
        model, x_val, y_val, layer_name=layer_name, class_names=class_names
    )
    df_interventions = analyze_interventions(
        model, x_val, y_val, layer_name=layer_name
    )

    df_post_weights.to_csv(out_path / "post_training_weights.csv", index=False)
    df_post_act.to_csv(out_path / "post_training_activations.csv", index=False)
    df_interventions.to_csv(out_path / "post_training_interventions.csv", index=False)
    np.save(out_path / "post_training_similarity.npy", sim_post)

    if class_rms_post is not None:
        class_rms_post.to_csv(out_path / "post_training_class_rms.csv")

    # Representation summary table for post-training
    rep_summary = build_representation_table(df_post_weights, df_post_act, df_interventions)
    rep_summary.to_csv(out_path / "post_training_representation_summary.csv", index=False)

    # Side-by-side comparison table
    comparison_df = pd.DataFrame({
        "filter": np.arange(len(df_pre_weights)),
        "init_peak_hz": df_pre_weights["peak_hz"],
        "post_peak_hz": df_post_weights["peak_hz"],
        "peak_shift_hz": df_post_weights.get("peak_shift_hz", np.zeros(len(df_pre_weights))),
        "init_bw_hz": df_pre_weights["bandwidth_3db_hz"],
        "post_bw_hz": df_post_weights["bandwidth_3db_hz"],
        "relative_l2_change": df_post_weights.get("relative_l2_change", np.zeros(len(df_pre_weights))),
        "weight_cosine_sim": df_post_weights.get("weight_cosine_sim", np.ones(len(df_pre_weights))),
        "pre_neg_energy_pct": df_pre_act["negative_energy_pct"],
        "post_neg_energy_pct": df_post_act["negative_energy_pct"],
        "ablation_acc_drop": df_interventions["accuracy_drop"],
        "ablation_macro_f1_drop": df_interventions.get("macro_f1_drop", df_interventions["accuracy_drop"]),
    })
    comparison_df.to_csv(out_path / "pre_vs_post_comparison.csv", index=False)

    print("\n==================================================")
    print("PRE vs POST DRIFT SUMMARY")
    print("==================================================")
    print(comparison_df.round(3).to_string(index=False))
    print(f"\nAll diagnostic files exported to: {out_path.resolve()}")

    return {
        "pre_weights": df_pre_weights,
        "post_weights": df_post_weights,
        "pre_activations": df_pre_act,
        "post_activations": df_post_act,
        "interventions": df_interventions,
        "comparison": comparison_df,
    }
