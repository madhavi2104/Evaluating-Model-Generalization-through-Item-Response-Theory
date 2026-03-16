#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    import timm
except ImportError:
    timm = None


TIMM_MODELS = {
    "beit_base_patch16_224",
    "convnext_base",
    "convnextv2_large",
    "convmixer_768_32",
    "swinv2_tiny_window16_256",
    "swinv2_base_window8_256",
    "coatnet_0_rw_224",
    "cait_m36_384",
    "eva_giant_patch14_224",
    "maxvit_tiny_tf_512",
    "vit_base_patch16_224",
}

TORCHVISION_MODELS = {
    "resnet18",
    "resnet50",
    "resnet101",
    "wide_resnet50_2",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b4",
    "densenet121",
    "convnext_tiny",
    "convnext_base",
    "vit_b_16",
    "swin_t",
    "mnasnet1_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Head-only adaptation of ImageNet-pretrained models on CIFAR-100."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name from timm or torchvision.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Path to CIFAR-100 root. Can also be set via THESIS_DATA_ROOT. "
            "Expected structure: <root>/train and <root>/test if using ImageFolder."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root output directory. Can also be set via THESIS_OUTPUT_ROOT. "
            "Outputs will be stored under training_protocols/head_only_cifar100/<model>."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs for the classifier head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for the head optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save-logits",
        action="store_true",
        help="Also save per-sample logits for the test set.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_data_root(arg_value: Path | None, repo_root: Path) -> Path:
    if arg_value is not None:
        return arg_value.resolve()

    env_value = os.environ.get("THESIS_DATA_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    return (repo_root / "data" / "external" / "CIFAR100_split").resolve()


def resolve_output_root(arg_value: Path | None, repo_root: Path) -> Path:
    if arg_value is not None:
        return arg_value.resolve()

    env_value = os.environ.get("THESIS_OUTPUT_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()

    return (repo_root / "results").resolve()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, test_transform


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected CIFAR-100 split dataset structure not found.\n"
            f"Missing train/test folders under: {data_root}\n"
            "Expected layout:\n"
            "  <data-root>/train/<class_name>/...\n"
            "  <data-root>/test/<class_name>/..."
        )

    train_transform, test_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def get_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


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

    raise ValueError(f"Unsupported torchvision head replacement for model: {model_name}")


def unfreeze_torchvision_head(model_name: str, model: nn.Module) -> None:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        for p in model.fc.parameters():
            p.requires_grad = True
        return

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Module):
            for p in classifier.parameters():
                p.requires_grad = True
            return

    if model_name.startswith("vit_") and hasattr(model, "heads"):
        for p in model.heads.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unsupported torchvision head unfreezing for model: {model_name}")


def build_torchvision_model(model_name: str, num_classes: int) -> nn.Module:
    if not hasattr(models, model_name):
        raise ValueError(f"Unknown torchvision model: {model_name}")

    model_fn = getattr(models, model_name)

    try:
        model = model_fn(weights="DEFAULT")
    except Exception:
        weights_enum_name = f"{model_name.capitalize()}_Weights"
        model = model_fn(weights=None)

    freeze_backbone(model)
    model = replace_torchvision_head(model_name, model, num_classes)
    unfreeze_torchvision_head(model_name, model)
    return model


def build_timm_model(model_name: str, num_classes: int) -> nn.Module:
    if timm is None:
        raise ImportError(
            "timm is not installed, but a timm model was requested."
        )

    model = timm.create_model(model_name, pretrained=True)
    freeze_backbone(model)
    model.reset_classifier(num_classes=num_classes)
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name in TIMM_MODELS:
        return build_timm_model(model_name, num_classes)

    if model_name in TORCHVISION_MODELS or hasattr(models, model_name):
        return build_torchvision_model(model_name, num_classes)

    raise ValueError(
        f"Unknown model '{model_name}'. Add it to the TIMM_MODELS or TORCHVISION_MODELS set."
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_logits: bool = False,
) -> tuple[float, float, list[dict], list[dict] | None]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    correctness_rows: list[dict] = []
    logits_rows: list[dict] | None = [] if save_logits else None
    global_idx = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        is_correct = (preds == labels)

        running_loss += loss.item() * images.size(0)
        correct += is_correct.sum().item()
        total += labels.size(0)

        preds_cpu = preds.cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        correct_cpu = is_correct.cpu().numpy()

        for i in range(len(labels_cpu)):
            correctness_rows.append(
                {
                    "item_id": global_idx,
                    "true_label": int(labels_cpu[i]),
                    "predicted_label": int(preds_cpu[i]),
                    "correct": int(correct_cpu[i]),
                }
            )
            global_idx += 1

        if save_logits and logits_rows is not None:
            outputs_cpu = outputs.cpu().numpy()
            batch_start = global_idx - len(labels_cpu)
            for i in range(len(labels_cpu)):
                logits_rows.append(
                    {
                        "item_id": batch_start + i,
                        "logits": outputs_cpu[i].tolist(),
                    }
                )

    eval_loss = running_loss / max(total, 1)
    eval_acc = correct / max(total, 1)
    return eval_loss, eval_acc, correctness_rows, logits_rows


def save_csv(rows: list[dict], output_file: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to save for {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = resolve_repo_root()
    data_root = resolve_data_root(args.data_root, repo_root)
    output_root = resolve_output_root(args.output_root, repo_root)

    run_dir = (
        output_root
        / "training_protocols"
        / "head_only_cifar100"
        / args.model
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model:        {args.model}", flush=True)
    print(f"Data root:    {data_root}", flush=True)
    print(f"Output dir:   {run_dir}", flush=True)
    print(f"Device:       {device}", flush=True)
    print(f"Epochs:       {args.epochs}", flush=True)
    print(f"Batch size:   {args.batch_size}", flush=True)

    train_loader, test_loader = build_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = build_model(args.model, num_classes=100).to(device)

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {num_trainable:,} / {num_total:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(get_trainable_parameters(model), lr=args.lr)

    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            save_logits=False,
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(epoch_row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}",
            flush=True,
        )

    test_loss, test_acc, correctness_rows, logits_rows = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        save_logits=args.save_logits,
    )

    torch.save(model.state_dict(), run_dir / "head_only_model.pth")
    save_csv(history, run_dir / "training_history.csv")
    save_csv(correctness_rows, run_dir / "binary_correctness.csv")

    metadata = {
        "protocol": "head_only_cifar100",
        "model": args.model,
        "num_classes": 100,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "image_size": args.image_size,
        "data_root": str(data_root),
        "output_dir": str(run_dir),
        "final_test_loss": test_loss,
        "final_test_acc": test_acc,
        "trainable_params_only_head": True,
    }
    save_json(metadata, run_dir / "run_metadata.json")

    if args.save_logits and logits_rows is not None:
        save_json({"rows": logits_rows}, run_dir / "test_logits.json")

    print(f"Saved model to:            {run_dir / 'head_only_model.pth'}", flush=True)
    print(f"Saved history to:          {run_dir / 'training_history.csv'}", flush=True)
    print(f"Saved binary correctness:  {run_dir / 'binary_correctness.csv'}", flush=True)
    print(f"Saved metadata to:         {run_dir / 'run_metadata.json'}", flush=True)
    if args.save_logits:
        print(f"Saved logits to:           {run_dir / 'test_logits.json'}", flush=True)


if __name__ == "__main__":
    main()
