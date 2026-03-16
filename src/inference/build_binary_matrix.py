#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "item_id",
    "item_path",
    "true_label_idx",
    "true_label_name",
    "correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-model binary correctness files into one IRT-ready "
            "binary response matrix with rows=items and columns=models."
        )
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing per-model prediction outputs for one "
            "dataset/regime. Example: results/predictions/CIFAR100/trained"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where merged outputs will be saved.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name, e.g. CIFAR100, ImageNet, ImageNet-C, Sketch.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        required=True,
        help="Regime name, e.g. zero_shot, trained, head_only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Require exact item alignment across all models. "
            "Fail if any model is missing items or has mismatched metadata."
        ),
    )
    return parser.parse_args()


def find_prediction_files(predictions_root: Path) -> list[Path]:
    files = sorted(predictions_root.glob("*/binary_correctness.csv"))
    return [f for f in files if f.is_file()]


def infer_model_name(file_path: Path) -> str:
    return file_path.parent.name


def validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {file_path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def load_one_model(file_path: Path) -> tuple[str, pd.DataFrame, dict]:
    model_name = infer_model_name(file_path)
    df = pd.read_csv(file_path)

    validate_columns(df, file_path)

    out = df[
        [
            "item_id",
            "item_path",
            "true_label_idx",
            "true_label_name",
            "correct",
        ]
    ].copy()

    out = out.rename(columns={"correct": model_name})
    out[model_name] = out[model_name].astype(int)

    info = {
        "model": model_name,
        "source_file": str(file_path),
        "num_rows": int(len(out)),
    }
    return model_name, out, info


def check_reference_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    model_name: str,
    strict: bool,
) -> None:
    ref_cols = ["item_id", "item_path", "true_label_idx", "true_label_name"]
    cand_cols = ["item_id", "item_path", "true_label_idx", "true_label_name"]

    if len(reference) != len(candidate):
        msg = (
            f"Row count mismatch for model '{model_name}': "
            f"reference has {len(reference)} rows, candidate has {len(candidate)} rows."
        )
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}", flush=True)

    merged = reference[ref_cols].merge(
        candidate[cand_cols],
        on=["item_id", "item_path", "true_label_idx", "true_label_name"],
        how="outer",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        n_bad = int((merged["_merge"] != "both").sum())
        msg = (
            f"Metadata alignment mismatch for model '{model_name}'. "
            f"{n_bad} rows differ in item identity metadata."
        )
        if strict:
            raise ValueError(msg)
        print(f"Warning: {msg}", flush=True)


def merge_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    merged = tables[0].copy()

    key_cols = ["item_id", "item_path", "true_label_idx", "true_label_name"]

    for table in tables[1:]:
        new_model_cols = [c for c in table.columns if c not in key_cols]
        merged = merged.merge(table, on=key_cols, how="outer")
        for col in new_model_cols:
            merged[col] = merged[col].astype("Int64")

    merged = merged.sort_values("item_id").reset_index(drop=True)
    return merged


def build_binary_only_matrix(merged: pd.DataFrame) -> pd.DataFrame:
    model_cols = [
        c for c in merged.columns
        if c not in ["item_id", "item_path", "true_label_idx", "true_label_name"]
    ]
    binary_only = merged[["item_id"] + model_cols].copy()
    return binary_only


def main() -> None:
    args = parse_args()

    predictions_root = args.predictions_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_root.exists():
        raise FileNotFoundError(f"Predictions root does not exist: {predictions_root}")

    prediction_files = find_prediction_files(predictions_root)
    if not prediction_files:
        raise FileNotFoundError(
            f"No per-model binary_correctness.csv files found under: {predictions_root}"
        )

    print(f"Found {len(prediction_files)} model prediction files.", flush=True)

    loaded_tables: list[pd.DataFrame] = []
    source_infos: list[dict] = []

    reference_df: pd.DataFrame | None = None
    seen_models: set[str] = set()

    for file_path in prediction_files:
        model_name, table, info = load_one_model(file_path)

        if model_name in seen_models:
            print(f"Skipping duplicate model directory: {model_name}", flush=True)
            continue

        seen_models.add(model_name)

        if reference_df is None:
            reference_df = table.copy()
        else:
            check_reference_alignment(
                reference=reference_df,
                candidate=table,
                model_name=model_name,
                strict=args.strict,
            )

        loaded_tables.append(table)
        source_infos.append(info)

        print(
            f"Loaded model={model_name} | rows={len(table)} | file={file_path}",
            flush=True,
        )

    if not loaded_tables:
        raise RuntimeError("No model prediction tables were loaded successfully.")

    merged = merge_tables(loaded_tables)

    model_cols = [
        c for c in merged.columns
        if c not in ["item_id", "item_path", "true_label_idx", "true_label_name"]
    ]

    missing_per_model = {
        col: int(merged[col].isna().sum())
        for col in model_cols
    }

    if args.strict and any(v > 0 for v in missing_per_model.values()):
        raise ValueError(
            "Missing model responses detected in strict mode after merge. "
            f"Missing counts: {missing_per_model}"
        )

    binary_only = build_binary_only_matrix(merged)

    matrix_with_metadata_file = output_dir / "binary_response_matrix_with_metadata.csv"
    merged.to_csv(matrix_with_metadata_file, index=False)

    binary_only_file = output_dir / "binary_response_matrix.csv"
    binary_only.to_csv(binary_only_file, index=False)

    item_metadata_file = output_dir / "item_metadata.csv"
    merged[["item_id", "item_path", "true_label_idx", "true_label_name"]].to_csv(
        item_metadata_file,
        index=False,
    )

    metadata = {
        "dataset": args.dataset,
        "regime": args.regime,
        "predictions_root": str(predictions_root),
        "num_models": len(model_cols),
        "num_items": int(len(merged)),
        "models": model_cols,
        "missing_values_per_model": missing_per_model,
        "strict_mode": args.strict,
        "source_files": source_infos,
        "outputs": {
            "binary_response_matrix": str(binary_only_file),
            "binary_response_matrix_with_metadata": str(matrix_with_metadata_file),
            "item_metadata": str(item_metadata_file),
        },
    }

    metadata_file = output_dir / "binary_response_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved binary matrix to:          {binary_only_file}", flush=True)
    print(f"Saved matrix with metadata to:   {matrix_with_metadata_file}", flush=True)
    print(f"Saved item metadata to:          {item_metadata_file}", flush=True)
    print(f"Saved merge metadata to:         {metadata_file}", flush=True)
    print(
        f"Final matrix shape: {len(merged)} items x {len(model_cols)} models",
        flush=True,
    )


if __name__ == "__main__":
    main()
