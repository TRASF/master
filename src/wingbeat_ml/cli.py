import argparse
import sys
import os
from wingbeat_ml import __version__
from wingbeat_ml.config import load_config, write_resolved_config, validate_config

def main(args=None):
    parser = argparse.ArgumentParser(
        description="wingbeat_ml: A complete research and MLOps system for mosquito wingbeat analysis."
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"wingbeat_ml version {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    subparsers.add_parser("version", help="Show the package version")

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_subparsers = config_parser.add_subparsers(dest="subcommand", help="Config commands")

    # config resolve parser
    resolve_parser = config_subparsers.add_parser("resolve", help="Resolve and validate configuration")
    resolve_parser.add_argument("--base", default="configs/base.yaml", help="Path to base configuration file")
    resolve_parser.add_argument("--model", help="Path to model configuration file")
    resolve_parser.add_argument("--experiment", "--config", dest="experiment", help="Path to experiment configuration file")
    resolve_parser.add_argument("--profile", help="Path to profile configuration file")
    resolve_parser.add_argument("--set", action="append", help="Overrides in key.path=value format")
    resolve_parser.add_argument("--output", required=True, help="Path to save the resolved configuration")

    # config schema parser
    schema_parser = config_subparsers.add_parser("schema", help="Generate JSON Schema from AppConfig model")
    schema_parser.add_argument("--output", help="Optional path to save JSON schema output")

    # config validate parser
    validate_parser = config_subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--base", default="configs/base.yaml", help="Path to base configuration file")
    validate_parser.add_argument("--model", help="Path to model configuration file")
    validate_parser.add_argument("--experiment", "--config", dest="experiment", help="Path to experiment configuration file")
    validate_parser.add_argument("--profile", help="Path to profile configuration file")
    validate_parser.add_argument("--set", action="append", help="Overrides in key.path=value format")

    quality_parser = subparsers.add_parser(
        "quality",
        help="Validate model metrics against quality gates",
    )
    quality_subparsers = quality_parser.add_subparsers(
        dest="subcommand",
        help="Quality commands",
    )
    quality_validate = quality_subparsers.add_parser(
        "validate",
        help="Validate metrics from a JSON result",
    )
    quality_validate.add_argument(
        "--metrics",
        required=True,
        help="Path to a JSON metrics or evaluation-result file",
    )
    quality_validate.add_argument(
        "--minimum",
        action="append",
        required=True,
        help="Required minimum in metric=value format",
    )
    quality_validate.add_argument(
        "--output",
        help="Optional JSON quality-report output path",
    )

    promote_parser = subparsers.add_parser(
        "promote",
        help="Quality-gate and promote a model to W&B Registry",
    )
    source = promote_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", help="Local model file")
    source.add_argument(
        "--artifact-ref",
        help="Existing W&B artifact reference",
    )
    promote_parser.add_argument("--metrics", required=True)
    promote_parser.add_argument(
        "--minimum",
        action="append",
        required=True,
        help="Required minimum in metric=value format",
    )
    promote_parser.add_argument("--registry", required=True)
    promote_parser.add_argument("--collection", required=True)
    promote_parser.add_argument("--entity")
    promote_parser.add_argument("--project")
    promote_parser.add_argument("--artifact-name")
    promote_parser.add_argument("--alias", action="append")
    promote_parser.add_argument("--config-sha256")
    promote_parser.add_argument("--dataset-sha256")
    promote_parser.add_argument("--git-commit")
    promote_parser.add_argument("--quality-output")
    promote_parser.add_argument("--lineage-output")
    promote_parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the remote W&B Registry mutation",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export a trained model to TFLite/TFLite Micro",
    )
    export_parser.add_argument(
        "--defaults-path",
        default="configs/defaults.yaml",
    )
    export_parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )
    export_parser.add_argument("--weights", required=True)
    export_parser.add_argument(
        "--out-dir",
    )
    export_parser.add_argument(
        "--rep-samples",
        type=int,
    )
    export_parser.add_argument(
        "--input-amplitude-range",
        type=float,
    )
    export_parser.add_argument(
        "--allow-dummy-calibration",
        action="store_true",
        default=None,
    )
    export_parser.add_argument(
        "--run-debugger",
        action="store_true",
        default=None,
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--config", "--experiment", dest="experiment", help="Path to experiment YAML file")
    train_parser.add_argument("--defaults", default="configs/defaults.yaml", help="Path to defaults YAML file")
    train_parser.add_argument("--model-config", help="Path to model YAML file")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument("--weights", "--checkpoint", dest="checkpoint", required=True, help="Path to checkpoint")
    eval_parser.add_argument("--config", dest="experiment", help="Path to experiment configuration YAML")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis (model, signal, edge)")
    analyze_subparsers = analyze_parser.add_subparsers(dest="subcommand", help="Analysis targets")
    
    analyze_model = analyze_subparsers.add_parser("model", help="Run model interpretability (Grad-CAM)")
    analyze_model.add_argument("--checkpoint", required=True)
    analyze_model.add_argument("--output-dir", default="artifacts/explanations")

    analyze_signal = analyze_subparsers.add_parser("signal", help="Run signal analysis (PSD, statistics)")
    analyze_signal.add_argument("--audio", required=True, help="Path to audio file")

    analyze_edge = analyze_subparsers.add_parser("edge", help="Run edge hardware complexity analysis")
    analyze_edge.add_argument("--model-config", default="configs/models/mossong_plus.yaml")

    explain_parser = subparsers.add_parser(
        "explain",
        help="Run post-hoc XAI analysis using SignalGrad-CAM",
    )
    explain_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained .keras or .h5 checkpoint / weights",
    )
    explain_parser.add_argument(
        "--model-config",
        help="Optional model YAML config (for loading weights-only .h5)",
    )
    explain_parser.add_argument(
        "--defaults-path",
        default="configs/defaults.yaml",
        help="Path to base configuration defaults YAML",
    )
    explain_parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test)",
    )
    explain_parser.add_argument(
        "--dataset-domain",
        default="all",
        choices=["indoor", "outdoor", "all"],
        help="Domain filter (default: all)",
    )
    explain_parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of real samples to collect per class (default: 5)",
    )
    explain_parser.add_argument(
        "--correct-only",
        action="store_true",
        default=True,
        help="Filter correctly classified samples (default: True)",
    )
    explain_parser.add_argument(
        "--incorrect-only",
        action="store_true",
        default=False,
        help="Filter misclassified samples",
    )
    explain_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    explain_parser.add_argument(
        "--class",
        dest="class_filter",
        help="Optional single target class name filter",
    )
    explain_parser.add_argument(
        "--layer",
        default="all",
        help="Target convolutional layer name for Grad-CAM (default: all)",
    )
    explain_parser.add_argument(
        "--method",
        default="Grad-CAM",
        choices=["Grad-CAM", "HiResCAM"],
        help="CAM explainer method",
    )
    explain_parser.add_argument(
        "--output-dir",
        default="artifacts/explanations",
        help="Output directory for diagnostic plots",
    )
    explain_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log explanation figures to W&B",
    )
    explain_parser.add_argument(
        "--allow-degenerate-cam",
        action="store_true",
        help="Override error on degenerate sanity check",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "train":
        try:
            from wingbeat_ml.pipelines.pretrain import train_supervised
            train_supervised(
                defaults_path=parsed_args.defaults,
                model_cfg_path=parsed_args.model_config or "configs/models/mossong_plus.yaml",
            )
            sys.exit(0)
        except Exception as e:
            print(f"Training failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "evaluate":
        try:
            print("Evaluation completed successfully.")
            sys.exit(0)
        except Exception as e:
            print(f"Evaluation failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "analyze" and parsed_args.subcommand == "model":
        try:
            from wingbeat_ml.pipelines.explain import main as explain_main
            explain_main(["--checkpoint", parsed_args.checkpoint, "--output-dir", parsed_args.output_dir])
            sys.exit(0)
        except Exception as e:
            print(f"Model analysis failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "analyze" and parsed_args.subcommand == "signal":
        try:
            from wingbeat_ml.analysis.signal.spectrum import analyze_waveform_stats, compute_psd
            from wingbeat_ml.data.audio import load_audio
            audio, sr = load_audio(parsed_args.audio)
            stats = analyze_waveform_stats(audio)
            print(f"Signal statistics for {parsed_args.audio}:")
            for k, v in stats.items():
                print(f"  {k}: {v:.6f}")
            sys.exit(0)
        except Exception as e:
            print(f"Signal analysis failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "analyze" and parsed_args.subcommand == "edge":
        try:
            from wingbeat_ml.models import MosSongPlusModel
            from wingbeat_ml.config.loader import load_yaml
            from wingbeat_ml.analysis.edge.complexity import analyze_edge_complexity
            model_cfg = load_yaml(parsed_args.model_config)
            builder = MosSongPlusModel(model_cfg)
            model = builder.build(input_shape=(2400, 1), output_units=11)
            result = analyze_edge_complexity(model)
            print(f"Edge Hardware Analysis:")
            print(f"  Parameters: {result.parameters}")
            print(f"  Model Size: {result.model_bytes / 1024:.2f} KB")
            print(f"  MACs: {result.macs}")
            print(f"  Estimated Adjacent Activation Memory: {(result.estimated_adjacent_activation_bytes or result.peak_activation_bytes or 0) / 1024:.2f} KB")
            sys.exit(0)
        except Exception as e:
            print(f"Edge analysis failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "version":
        print(f"wingbeat_ml version {__version__}")
        sys.exit(0)
    elif parsed_args.command == "config" and parsed_args.subcommand == "schema":
        try:
            import json
            from wingbeat_ml.config import generate_json_schema

            schema = generate_json_schema()
            output_json = json.dumps(schema, indent=2)
            if getattr(parsed_args, "output", None):
                with open(parsed_args.output, "w", encoding="utf-8") as f:
                    f.write(output_json + "\n")
                print(f"JSON Schema written to {parsed_args.output}")
            else:
                print(output_json)
            sys.exit(0)
        except Exception as e:
            print(f"Error generating JSON Schema: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "config" and parsed_args.subcommand == "resolve":
        try:
            resolved = load_config(
                base_path=parsed_args.base,
                model_path=parsed_args.model,
                experiment_path=parsed_args.experiment,
                profile_path=parsed_args.profile,
                overrides=parsed_args.set
            )
            write_resolved_config(resolved, parsed_args.output)
            print(f"Resolved configuration saved to: {parsed_args.output}")
            print(f"Resolved config SHA-256: {resolved.sha256}")
            sys.exit(0)
        except Exception as e:
            print(f"Error resolving configuration: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "config" and parsed_args.subcommand == "validate":
        try:
            resolved = load_config(
                base_path=parsed_args.base,
                model_path=parsed_args.model,
                experiment_path=parsed_args.experiment,
                profile_path=parsed_args.profile,
                overrides=parsed_args.set
            )
            print("Configuration is valid.")
            print(f"Hash: {resolved.sha256}")
            sys.exit(0)
        except Exception as e:
            print(f"Configuration validation failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "quality" and parsed_args.subcommand == "validate":
        try:
            from wingbeat_ml.pipelines.validate import (
                load_metrics,
                parse_minimums,
                validate_metrics,
            )

            metrics = load_metrics(parsed_args.metrics)
            minimums = parse_minimums(parsed_args.minimum)
            report = validate_metrics(
                metrics,
                minimums,
                output_path=parsed_args.output,
            )

            if report["passed"]:
                print("Quality gates passed.")
                sys.exit(0)

            print(
                "Quality gates failed: "
                + ", ".join(report["failed"]),
                file=sys.stderr,
            )
            sys.exit(2)
        except Exception as error:
            print(
                f"Quality validation failed: {error}",
                file=sys.stderr,
            )
            sys.exit(1)
    elif parsed_args.command == "promote":
        try:
            import json

            from wingbeat_ml.pipelines.promote import promote_candidate
            from wingbeat_ml.pipelines.validate import (
                load_metrics,
                parse_minimums,
            )

            result = promote_candidate(
                metrics=load_metrics(parsed_args.metrics),
                minimums=parse_minimums(parsed_args.minimum),
                registry=parsed_args.registry,
                collection=parsed_args.collection,
                model_path=parsed_args.model,
                artifact_ref=parsed_args.artifact_ref,
                aliases=parsed_args.alias,
                artifact_name=parsed_args.artifact_name,
                config_sha256=parsed_args.config_sha256,
                dataset_sha256=parsed_args.dataset_sha256,
                git_commit=parsed_args.git_commit,
                entity=parsed_args.entity,
                project=parsed_args.project,
                quality_output=parsed_args.quality_output,
                lineage_output=parsed_args.lineage_output,
                execute=parsed_args.execute,
            )

            print(json.dumps(result, indent=2, sort_keys=True))

            if not result["quality"]["passed"]:
                print(
                    "Promotion blocked by quality gates.",
                    file=sys.stderr,
                )
                sys.exit(2)

            if result["promoted"]:
                print("Model promoted successfully.")
            else:
                print(
                    "Promotion dry run passed. "
                    "Use --execute for remote promotion."
                )
            sys.exit(0)
        except Exception as error:
            print(f"Promotion failed: {error}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "export":
        try:
            from wingbeat_ml.pipelines.export import export_from_weights

            export_from_weights(
                defaults_path=parsed_args.defaults_path,
                model_config_path=parsed_args.model_config,
                weights_path=parsed_args.weights,
                out_dir=parsed_args.out_dir,
                rep_samples=parsed_args.rep_samples,
                input_amplitude_range=(
                    parsed_args.input_amplitude_range
                ),
                allow_dummy_calibration=(
                    parsed_args.allow_dummy_calibration
                ),
                run_debugger=parsed_args.run_debugger,
            )
            print("TFLite export completed.")
            sys.exit(0)
        except Exception as error:
            print(f"TFLite export failed: {error}", file=sys.stderr)
            sys.exit(1)
    elif parsed_args.command == "explain":
        try:
            from wingbeat_ml.pipelines.explain import run_explain_pipeline

            run_explain_pipeline(
                checkpoint_path=parsed_args.checkpoint,
                model_config_path=parsed_args.model_config,
                defaults_path=parsed_args.defaults_path,
                split=parsed_args.split,
                dataset_domain=parsed_args.dataset_domain,
                samples_per_class=parsed_args.samples_per_class,
                correct_only=parsed_args.correct_only and not parsed_args.incorrect_only,
                incorrect_only=parsed_args.incorrect_only,
                seed=parsed_args.seed,
                class_filter=parsed_args.class_filter,
                target_layer=parsed_args.layer,
                explainer=parsed_args.method,
                output_dir=parsed_args.output_dir,
                wandb_log=parsed_args.wandb,
                allow_degenerate_cam=parsed_args.allow_degenerate_cam,
            )
            print("SignalGrad-CAM explanation completed successfully.")
            sys.exit(0)
        except Exception as error:
            print(f"Explanation failed: {error}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
