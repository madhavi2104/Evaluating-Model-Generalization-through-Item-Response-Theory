#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DATASET_SUBDIRS = {
    "Sketch": "ImageNet_Sketch_SPLIT",
    "ImageNet-C": "ImageNet-C_split",
    "CIFAR100": "CIFAR100_split",
}

CUSTOM_SETTINGS = {
    "vit_b_16": {"batch_size": 16, "workers": 4},
    "convnext_large": {"batch_size": 12, "workers": 2},
    "beit_base_patch16_224": {"batch_size": 8, "workers": 2},
    "mnasnet1_0": {"batch_size": 16, "workers": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch torchvision training with repo-safe paths."
    )
    parser.add_argument("model", type=str, help="Torchvision model name")
    parser.add_argument(
        "dataset",
        type=str,
        choices=sorted(DATASET_SUBDIRS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing dataset folders. "
            "Can also be set via THESIS_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory for training outputs. "
            "Can also be set via THESIS_OUTPUT_ROOT."
        ),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=None,
        help="Optional explicit path to the torchvision training script.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_root(arg_value: Path | None, repo_root: Path) -> Path:
    if arg_value is not None:
        return arg_value.resolve()

    env_value = os.environ.get("THESIS_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    return (repo_root / "data" / "external").resolve()


def resolve_output_root(arg_value: Path | None, repo_root: Path) -> Path:
    if arg_value is not None:
        return arg_value.resolve()

    env_value = os.environ.get("THESIS_OUTPUT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    return (repo_root / "results").resolve()


def resolve_train_script(arg_value: Path | None, repo_root: Path) -> Path:
    if arg_value is not None:
        return arg_value.resolve()

    return (
        repo_root / "src" / "training" / "backends" / "torchvision" / "train.py"
    ).resolve()


def resolve_dataset_path(data_root: Path, dataset: str) -> Path:
    return (data_root / DATASET_SUBDIRS[dataset]).resolve()


def resolve_output_dir(output_root: Path, dataset: str, model: str) -> Path:
    return (output_root / "training" / dataset / model).resolve()


def resolve_hyperparameters(model: str, dataset: str) -> tuple[int, int]:
    batch_size = 32
    workers = 8

    if dataset == "CIFAR100":
        batch_size = 64
        workers = 4

    if model in CUSTOM_SETTINGS:
        batch_size = CUSTOM_SETTINGS[model]["batch_size"]
        workers = CUSTOM_SETTINGS[model]["workers"]

    return batch_size, workers


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()

    data_root = resolve_data_root(args.data_root, repo_root)
    output_root = resolve_output_root(args.output_root, repo_root)
    train_script = resolve_train_script(args.train_script, repo_root)

    data_path = resolve_dataset_path(data_root, args.dataset)
    out_dir = resolve_output_dir(output_root, args.dataset, args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {data_path}\n"
            f"Set --data-root or THESIS_DATA_ROOT correctly."
        )

    if not train_script.exists():
        raise FileNotFoundError(
            f"Training script not found: {train_script}\n"
            f"Set --train-script explicitly or place the file in the repo path."
        )

    batch_size, workers = resolve_hyperparameters(args.model, args.dataset)

    print(
        f"Starting torchvision wrapper for model={args.model} | dataset={args.dataset}",
        flush=True,
    )
    print(f"Resolved data path:   {data_path}", flush=True)
    print(f"Resolved output dir:  {out_dir}", flush=True)
    print(f"Resolved train file:  {train_script}", flush=True)
    print(
        f"Resolved settings -> batch_size={batch_size}, workers={workers}",
        flush=True,
    )

    cmd = [
        sys.executable,
        str(train_script),
        "--model",
        args.model,
        "--data-path",
        str(data_path),
        "--output-dir",
        str(out_dir),
        "--epochs",
        "300",
        "--checkpoint-freq",
        "50",
        "--early-stop-patience",
        "30",
        "--opt",
        "adamw",
        "--lr",
        "0.001",
        "--lr-scheduler",
        "cosineannealinglr",
        "--lr-warmup-epochs",
        "5",
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--auto-augment",
        "ta_wide",
    ]

    print("Executing command:", flush=True)
    print(" ".join(cmd), flush=True)

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
