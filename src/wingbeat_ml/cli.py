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
        default="quantization_output",
    )
    export_parser.add_argument(
        "--rep-samples",
        type=int,
        default=500,
    )
    export_parser.add_argument(
        "--input-amplitude-range",
        type=float,
    )
    export_parser.add_argument(
        "--allow-dummy-calibration",
        action="store_true",
    )
    export_parser.add_argument(
        "--run-debugger",
        action="store_true",
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "version":
        print(f"wingbeat_ml version {__version__}")
        sys.exit(0)
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
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
