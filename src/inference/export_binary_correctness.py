#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import timm
except ImportError:
    timm = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_ZERO_SHOT_DATASETS = {"ImageNet", "ImageNet-C", "Sketch", "ImageNet-Sketch"}
DEFAULT_TRAINED_DATASETS = {"ImageNet", "ImageNet-C", "Sketch", "ImageNet-Sketch", "CIFAR100"}

# Models explicitly known from your project
KNOWN_TIMM_MODELS = {
    "beit_base_patch16_224",
    "botnet26t_256",
    "cait_m36_384",
    "coatnet_0_rw_224",
    "coat_lite_tiny",
    "convmixer_768_32",
    "convnext_base",
    "convnextv2_tiny",
    "convnextv2_large",
    "crossvit_18_dagger_408",
    "darknet53",
    "davit_base",
    "deit_base_patch16_224",
    "densenet201",
    "dpn107",
    "efficientformer_l1",
    "ese_vovnet19b_dw",
    "eva_giant_patch14_224",
    "hrnet_w18",
    "inception_resnet_v2",
    "levit_384",
    "maxvit_tiny_tf_512",
    "mixer_b16_224",
    "mixnet_l",
    "mobilevit_s",
    "nfnet_f0",
    "regnetz_b16",
    "repvgg_b1g4",
    "resmlp_24_224",
    "swinv2_base_window8_256",
    "swinv2_tiny_window16_256",
    "vit_base_patch16_224",
    "volo_d1_224",
    "xception41",
    "xcit_small_12_p8_224",
}

SPECIAL_IMAGE_SIZES = {
    "inception_v3": 299,
    "googlenet": 224,
    "levit_384": 224,
}

DATASET_DEFAULT_SPLITS = {
    "ImageNet": "val",
    "ImageNet-C": "test",
    "Sketch": "test",
    "ImageNet-Sketch": "test",
    "CIFAR100": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-item predictions and binary correctness for zero-shot or trained evaluation."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Model name.")
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
        help=(
            "Evaluation regime. "
            "'zero_shot' = ImageNet-pretrained weights, "
            "'trained' = dataset-specific trained checkpoint, "
            "'head_only' = head-only CIFAR100 checkpoint."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Path to dataset root. Can also be set via THESIS_DATA_ROOT. "
            "Expected structure: <root>/<split>/<class_name>/..."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory for outputs. Can also be set via THESIS_OUTPUT_ROOT. "
            "Outputs go under results/predictions/<dataset>/<regime>/<model>/ by default."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint for trained or head_only evaluation. "
            "Required for trained/head_only. Optional for zero_shot."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "timm", "torchvision"],
        help="Model backend. 'auto' tries to infer it.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split directory name. Defaults depend on dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
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
        help="Optional override for input image size.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Also save logits JSON for each sample.",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Use strict=True when loading checkpoints.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def resolve_split(dataset: str, split: str | None) -> str:
    if split is not None:
        return split
    return DATASET_DEFAULT_SPLITS.get(dataset, "test")


def infer_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend

    if model_name in KNOWN_TIMM_MODELS:
        return "timm"

    if hasattr(models, model_name):
        return "torchvision"

    if timm is not None:
        try:
            available = timm.list_models(pretrained=False)
            if model_name in available:
                return "timm"
        except Exception:
            pass

    raise ValueError(
        f"Could not infer backend for model '{model_name}'. "
        "Pass --backend timm or --backend torchvision explicitly."
    )


def infer_image_size(model_name: str, override: int | None) -> int:
    if override is not None:
        return override

    if model_name in SPECIAL_IMAGE_SIZES:
        return SPECIAL_IMAGE_SIZES[model_name]

    match = re.search(r"_(\d{3})$", model_name)
    if match:
        return int(match.group(1))

    return 224


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_dataset(data_root: Path, dataset: str, split: str, image_size: int) -> datasets.ImageFolder:
    if data_root.name == split:
        split_root = data_root
    else:
        split_root = data_root / split

    if not split_root.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_root}\n"
            f"Expected layout: <data-root>/{split}/<class_name>/..."
        )

    dataset_obj = datasets.ImageFolder(
        root=str(split_root),
        transform=build_transform(image_size),
    )
    return dataset_obj


def build_dataloader(dataset_obj: datasets.ImageFolder, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def replace_torchvision_head(model_name: str, model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier

        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return model

        if isinstance(classifier, nn.Sequential):
            layers = list(classifier.children())
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    in_features = layers[idx].in_features
                    layers[idx] = nn.Linear(in_features, num_classes)
                    model.classifier = nn.Sequential(*layers)
                    return model

    if model_name.startswith("vit_") and hasattr(model, "heads"):
        if hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            return model

    raise ValueError(f"Unsupported torchvision classifier replacement for model: {model_name}")


def build_torchvision_model(model_name: str, regime: str, num_classes: int, pretrained: bool) -> nn.Module:
    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")

    model_fn = getattr(models, model_name)

    if regime == "zero_shot":
        if num_classes != 1000:
            raise ValueError(
                "Zero-shot torchvision evaluation expects ImageNet-compatible label space (1000 classes). "
                "For CIFAR100 use head_only or trained evaluation."
            )
        model = model_fn(weights="DEFAULT")
        return model

    if regime in {"trained", "head_only"}:
        model = model_fn(weights=None)
        model = replace_torchvision_head(model_name, model, num_classes)
        return model

    raise ValueError(f"Unsupported regime: {regime}")


def build_timm_model(model_name: str, regime: str, num_classes: int, pretrained: bool) -> nn.Module:
    if timm is None:
        raise ImportError("timm is not installed, but a timm model was requested.")

    if regime == "zero_shot":
        if num_classes != 1000:
            raise ValueError(
                "Zero-shot timm evaluation expects ImageNet-compatible label space (1000 classes). "
                "For CIFAR100 use head_only or trained evaluation."
            )
        return timm.create_model(model_name, pretrained=True)

    if regime in {"trained", "head_only"}:
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    raise ValueError(f"Unsupported regime: {regime}")


def extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Could not extract a model state_dict from checkpoint.")


def strip_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["module.", "_orig_mod."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint_path: Path,
    strict: bool,
) -> tuple[list[str], list[str]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(ckpt)
    state_dict = strip_prefixes(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def build_model(
    model_name: str,
    backend: str,
    regime: str,
    dataset_name: str,
    num_classes: int,
) -> nn.Module:
    if backend == "torchvision":
        return build_torchvision_model(
            model_name=model_name,
            regime=regime,
            num_classes=num_classes,
            pretrained=(regime == "zero_shot"),
        )

    if backend == "timm":
        return build_timm_model(
            model_name=model_name,
            regime=regime,
            num_classes=num_classes,
            pretrained=(regime == "zero_shot"),
        )

    raise ValueError(f"Unsupported backend: {backend}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    dataset_obj: datasets.ImageFolder,
    split_root: Path,
    device: torch.device,
    save_logits: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, float]:
    model.eval()

    rows: list[dict[str, Any]] = []
    logits_rows: list[dict[str, Any]] | None = [] if save_logits else None

    class_names = dataset_obj.classes
    samples = dataset_obj.samples

    total_correct = 0
    total_seen = 0
    global_idx = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        elif hasattr(outputs, "logits"):
            outputs = outputs.logits

        preds = outputs.argmax(dim=1)
        correct = (preds == labels)

        preds_cpu = preds.cpu().tolist()
        labels_cpu = labels.cpu().tolist()
        correct_cpu = correct.cpu().tolist()

        if save_logits and logits_rows is not None:
            outputs_cpu = outputs.cpu().tolist()
        else:
            outputs_cpu = None

        for i in range(len(labels_cpu)):
            item_path, _ = samples[global_idx]
            rel_path = os.path.relpath(item_path, split_root)

            row = {
                "item_id": global_idx,
                "item_path": rel_path,
                "true_label_idx": int(labels_cpu[i]),
                "true_label_name": class_names[labels_cpu[i]],
                "predicted_label_idx": int(preds_cpu[i]),
                "predicted_label_name": class_names[preds_cpu[i]] if preds_cpu[i] < len(class_names) else str(preds_cpu[i]),
                "correct": int(correct_cpu[i]),
            }
            rows.append(row)

            if save_logits and logits_rows is not None and outputs_cpu is not None:
                logits_rows.append(
                    {
                        "item_id": global_idx,
                        "item_path": rel_path,
                        "logits": outputs_cpu[i],
                    }
                )

            total_correct += int(correct_cpu[i])
            total_seen += 1
            global_idx += 1

    accuracy = total_correct / max(total_seen, 1)
    return rows, logits_rows, accuracy


def save_csv(rows: list[dict[str, Any]], output_file: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to save for {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Any, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()

    repo_root = resolve_repo_root()
    data_root_base = resolve_data_root(args.data_root, repo_root)
    output_root = resolve_output_root(args.output_root, repo_root)

    split = resolve_split(args.dataset, args.split)

    # Support both passing the dataset root directly or a parent data root
    dataset_root = data_root_base
    if not (dataset_root / split).exists():
        candidate = data_root_base / args.dataset
        if (candidate / split).exists():
            dataset_root = candidate

    image_size = infer_image_size(args.model, args.image_size)
    dataset_obj = load_dataset(dataset_root, args.dataset, split, image_size)
    loader = build_dataloader(dataset_obj, args.batch_size, args.workers)

    backend = infer_backend(args.model, args.backend)

    if args.regime == "zero_shot":
        if args.dataset == "CIFAR100":
            raise ValueError(
                "Strict zero-shot CIFAR100 evaluation is not label-compatible with ImageNet classifier heads. "
                "Use regime='head_only' for the frozen-backbone adapted protocol, or regime='trained' with a checkpoint."
            )
        num_classes = 1000
    else:
        num_classes = len(dataset_obj.classes)

    if args.regime in {"trained", "head_only"} and args.checkpoint is None:
        raise ValueError(
            f"Regime '{args.regime}' requires --checkpoint."
        )

    model = build_model(
        model_name=args.model,
        backend=backend,
        regime=args.regime,
        dataset_name=args.dataset,
        num_classes=num_classes,
    )

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []

    if args.checkpoint is not None:
        missing_keys, unexpected_keys = load_checkpoint_into_model(
            model=model,
            checkpoint_path=args.checkpoint.resolve(),
            strict=args.strict_load,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_dir = (
        output_root
        / "predictions"
        / args.dataset
        / args.regime
        / args.model
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    split_root = dataset_root / split
    rows, logits_rows, accuracy = evaluate_model(
        model=model,
        loader=loader,
        dataset_obj=dataset_obj,
        split_root=split_root,
        device=device,
        save_logits=args.save_logits,
    )

    save_csv(rows, run_dir / "binary_correctness.csv")

    metadata = {
        "model": args.model,
        "backend": backend,
        "dataset": args.dataset,
        "regime": args.regime,
        "split": split,
        "data_root": str(dataset_root),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "image_size": image_size,
        "num_classes": num_classes,
        "num_items": len(rows),
        "accuracy": accuracy,
        "missing_checkpoint_keys": missing_keys,
        "unexpected_checkpoint_keys": unexpected_keys,
    }
    save_json(metadata, run_dir / "evaluation_metadata.json")

    if args.save_logits and logits_rows is not None:
        save_json({"rows": logits_rows}, run_dir / "logits.json")

    print(f"Saved binary correctness to: {run_dir / 'binary_correctness.csv'}", flush=True)
    print(f"Saved metadata to:          {run_dir / 'evaluation_metadata.json'}", flush=True)
    print(f"Accuracy: {accuracy:.4f}", flush=True)
    if args.save_logits:
        print(f"Saved logits to:           {run_dir / 'logits.json'}", flush=True)
    if missing_keys:
        print(f"Missing keys on load:      {missing_keys}", flush=True)
    if unexpected_keys:
        print(f"Unexpected keys on load:   {unexpected_keys}", flush=True)


if __name__ == "__main__":
    main()
