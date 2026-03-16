#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run binary response matrix merging for one or more dataset/regime combinations."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Datasets to process, e.g. ImageNet ImageNet-C Sketch CIFAR100",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        required=True,
        help="Regimes to process, e.g. zero_shot trained head_only",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root results directory containing predictions/ and response_matrices/.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Pass strict mode to the merge script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining merges even if one fails.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    merge_script = repo_root / "src" / "inference" / "build_binary_matrix.py"
    if not merge_script.exists():
        raise FileNotFoundError(f"Merge script not found: {merge_script}")

    results_root = args.results_root.resolve()
    summary_records: list[dict] = []

    for dataset in args.datasets:
        for regime in args.regimes:
            predictions_root = results_root / "predictions" / dataset / regime
            output_dir = results_root / "response_matrices" / dataset / regime

            cmd = [
                sys.executable,
                str(merge_script),
                "--predictions-root",
                str(predictions_root),
                "--output-dir",
                str(output_dir),
                "--dataset",
                dataset,
                "--regime",
                regime,
            ]

            if args.strict:
                cmd.append("--strict")

            print(f"\n=== Merging dataset={dataset} | regime={regime} ===", flush=True)
            print(" ".join(cmd), flush=True)

            record = {
                "dataset": dataset,
                "regime": regime,
                "predictions_root": str(predictions_root),
                "output_dir": str(output_dir),
                "status": "pending",
            }

            if args.dry_run:
                record["status"] = "dry_run"
                summary_records.append(record)
                continue

            if not predictions_root.exists():
                record["status"] = "missing_predictions_root"
                summary_records.append(record)
                message = f"Predictions root does not exist: {predictions_root}"
                if args.continue_on_error:
                    print(f"Warning: {message}", flush=True)
                    continue
                raise FileNotFoundError(message)

            try:
                subprocess.run(cmd, check=True)
                record["status"] = "success"
            except subprocess.CalledProcessError as e:
                record["status"] = "failed"
                record["returncode"] = e.returncode
                if not args.continue_on_error:
                    summary_records.append(record)
                    summary_file = results_root / "response_matrices" / "merge_summary.json"
                    summary_file.parent.mkdir(parents=True, exist_ok=True)
                    with summary_file.open("w") as f:
                        json.dump(summary_records, f, indent=2)
                    raise

            summary_records.append(record)

    summary_file = results_root / "response_matrices" / "merge_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w") as f:
        json.dump(summary_records, f, indent=2)

    print(f"\nSaved merge summary to: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
