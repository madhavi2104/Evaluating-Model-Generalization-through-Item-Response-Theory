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
            "Run batch evaluation over a list of models and export per-model "
            "binary correctness files."
        )
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=Path("configs/models/models.txt"),
        help="Path to text file containing one model name per line.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g. ImageNet, ImageNet-C, Sketch, ImageNet-Sketch, CIFAR100.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        required=True,
        choices=["zero_shot", "trained", "head_only"],
        help="Evaluation regime.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to dataset root or split root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory where results will be written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split override, e.g. test or val.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing per-model checkpoints for trained/head_only runs. "
            "Expected structure depends on --checkpoint-pattern."
        ),
    )
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="{checkpoint_root}/{dataset}/{model}/checkpoint.pth",
        help=(
            "Pattern used to construct checkpoint paths. Available fields: "
            "{checkpoint_root}, {dataset}, {model}, {regime}"
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "timm", "torchvision"],
        help="Backend override passed to export_binary_correctness.py.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional image size override.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Also save logits for each model.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Pass strict checkpoint loading to the exporter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating remaining models if one fails.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_models(models_file: Path) -> list[str]:
    if not models_file.exists():
        raise FileNotFoundError(f"Models file not found: {models_file}")

    models: list[str] = []
    with models_file.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            models.append(line)

    if not models:
        raise ValueError(f"No models found in: {models_file}")

    return models


def build_checkpoint_path(
    pattern: str,
    checkpoint_root: Path,
    dataset: str,
    regime: str,
    model: str,
) -> Path:
    checkpoint_str = pattern.format(
        checkpoint_root=str(checkpoint_root),
        dataset=dataset,
        regime=regime,
        model=model,
    )
    return Path(checkpoint_str).resolve()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    exporter = repo_root / "src" / "inference" / "export_binary_correctness.py"
    if not exporter.exists():
        raise FileNotFoundError(f"Exporter script not found: {exporter}")

    models_file = args.models_file.resolve()
    models = load_models(models_file)

    summary_dir = (
        args.output_root.resolve()
        / "predictions"
        / args.dataset
        / args.regime
    )
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict] = []
    failures = 0

    for model in models:
        cmd = [
            sys.executable,
            str(exporter),
            "--model",
            model,
            "--dataset",
            args.dataset,
            "--regime",
            args.regime,
            "--data-root",
            str(args.data_root.resolve()),
            "--output-root",
            str(args.output_root.resolve()),
            "--backend",
            args.backend,
            "--batch-size",
            str(args.batch_size),
            "--workers",
            str(args.workers),
        ]

        if args.split is not None:
            cmd.extend(["--split", args.split])

        if args.image_size is not None:
            cmd.extend(["--image-size", str(args.image_size)])

        if args.save_logits:
            cmd.append("--save-logits")

        if args.strict_load:
            cmd.append("--strict-load")

        checkpoint_path = None
        if args.regime in {"trained", "head_only"}:
            if args.checkpoint_root is None:
                raise ValueError(
                    f"--checkpoint-root is required for regime '{args.regime}'."
                )

            checkpoint_path = build_checkpoint_path(
                pattern=args.checkpoint_pattern,
                checkpoint_root=args.checkpoint_root.resolve(),
                dataset=args.dataset,
                regime=args.regime,
                model=model,
            )
            cmd.extend(["--checkpoint", str(checkpoint_path)])

        print(f"\n=== Evaluating model: {model} ===", flush=True)
        print(" ".join(cmd), flush=True)

        record = {
            "model": model,
            "dataset": args.dataset,
            "regime": args.regime,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "status": "pending",
        }

        if args.dry_run:
            record["status"] = "dry_run"
            run_records.append(record)
            continue

        try:
            subprocess.run(cmd, check=True)
            record["status"] = "success"
        except subprocess.CalledProcessError as e:
            record["status"] = "failed"
            record["returncode"] = e.returncode
            failures += 1

            if not args.continue_on_error:
                run_records.append(record)
                summary_file = summary_dir / "batch_evaluation_summary.json"
                with summary_file.open("w") as f:
                    json.dump(run_records, f, indent=2)
                raise

        run_records.append(record)

    summary_file = summary_dir / "batch_evaluation_summary.json"
    with summary_file.open("w") as f:
        json.dump(run_records, f, indent=2)

    print(f"\nSaved batch summary to: {summary_file}", flush=True)
    print(
        f"Finished batch evaluation: {len(models) - failures} succeeded, {failures} failed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
