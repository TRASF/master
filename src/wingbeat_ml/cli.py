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
    else:
        parser.print_help()
        sys.exit(0)

if __name__ == "__main__":
    main()
