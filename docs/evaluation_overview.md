# Evaluation and Binary Response Matrix Pipeline

This document describes how trained or pretrained models are evaluated and how their outputs are converted into the binary response matrices used for IRT analysis.

## Overview

The repository supports two main evaluation regimes:

1. **zero-shot evaluation**
2. **trained evaluation**

It also supports a separate CIFAR-100-specific protocol:

3. **head-only evaluation**

The goal of this stage is to move from model weights to **per-item binary correctness outputs**, and then from those per-model outputs to a **merged binary response matrix** suitable for IRT.

## Evaluation regimes

### 1. Zero-shot evaluation

In the zero-shot regime, models are evaluated using their **ImageNet-pretrained weights**.

This means:

- the model is loaded with pretrained ImageNet weights
- the model is evaluated directly on the test split of the target dataset
- no dataset-specific checkpoint is loaded
- the output is a per-item binary correctness file for each model

This regime is intended for datasets that are label-compatible with the ImageNet classifier head, such as:

- **ImageNet**
- **ImageNet-C**
- **ImageNet-Sketch / Sketch**

### Important note on CIFAR-100

Strict zero-shot evaluation is **not label-compatible** with CIFAR-100, because the original ImageNet classifier head predicts 1000 ImageNet classes, not 100 CIFAR classes.

For CIFAR-100, the repository therefore uses a separate **head-only adaptation protocol** instead of direct zero-shot evaluation.

### 2. Trained evaluation

In the trained regime, models are evaluated using **dataset-specific trained weights**.

This means:

- the correct model architecture is instantiated
- a trained checkpoint is loaded
- the model is evaluated on the test split of the same dataset it was trained on
- the output is a per-item binary correctness file for each model

This is the standard evaluation route for fully trained or fine-tuned models.

### 3. Head-only CIFAR-100 evaluation

For CIFAR-100, the repository also includes a separate protocol based on:

- ImageNet-pretrained initialization
- classifier replacement for CIFAR-100
- frozen backbone
- training only the classifier head
- evaluation on the CIFAR-100 test split

This protocol is called **head-only** in the repository.

It is distinct from both strict zero-shot evaluation and full dataset-specific training.

## Step 1: Export per-model binary correctness

The core evaluation script is:

```text
src/inference/export_binary_correctness.py
```

This script evaluates one model on one dataset under one regime and writes its per-item results.

### Output structure

For each model, the exporter writes:

```text
results/
  predictions/
    <dataset>/
      <regime>/
        <model>/
          binary_correctness.csv
          evaluation_metadata.json
          logits.json   # optional
```

### Contents of `binary_correctness.csv`

Each row corresponds to one evaluated item and includes:

- `item_id`
- `item_path`
- `true_label_idx`
- `true_label_name`
- `predicted_label_idx`
- `predicted_label_name`
- `correct`

The `correct` column contains the binary response used later for IRT:

- `1` if the prediction is correct
- `0` if the prediction is incorrect

## Step 2: Run evaluation for many models

The batch wrapper for evaluation is:

```text
scripts/run_batch_evaluation.py
```

This wrapper:

- reads the model list from `configs/models/models.txt`
- loops over all models
- calls `export_binary_correctness.py`
- writes one evaluation folder per model
- saves a batch summary JSON

### Model list

The file:

```text
configs/models/models.txt
```

is used as the canonical model inventory for batch evaluation.

Each line should contain one model name.

## Step 3: Merge per-model outputs into one binary response matrix

Once all models for a given dataset and regime have been evaluated, their per-model correctness files are merged into one wide matrix.

The core merge script is:

```text
src/inference/build_binary_matrix.py
```

This script reads all files of the form:

```text
results/predictions/<dataset>/<regime>/<model>/binary_correctness.csv
```

and merges them into a single matrix with:

- **rows = items**
- **columns = models**
- **entries = 0 or 1**

This is the main input format used for IRT.

### Merge outputs

The merge step writes:

```text
results/
  response_matrices/
    <dataset>/
      <regime>/
        binary_response_matrix.csv
        binary_response_matrix_with_metadata.csv
        item_metadata.csv
        binary_response_metadata.json
```

### Main output files

#### `binary_response_matrix.csv`

This is the main IRT input.

It contains:

- one row per item
- one column per model
- binary values `0` or `1`

The first column is `item_id`.

#### `binary_response_matrix_with_metadata.csv`

This file contains the same binary matrix, but also preserves item metadata such as:

- `item_path`
- `true_label_idx`
- `true_label_name`

This is useful for debugging, PCA analysis, and linking item-level results back to the original images.

#### `item_metadata.csv`

This contains only the item metadata once, separated from the binary model-response matrix.

#### `binary_response_metadata.json`

This contains metadata about the merge process, including:

- dataset
- regime
- source files used
- number of items
- number of models
- missing-value summary

## Step 4: Run merge for many dataset/regime combinations

The batch wrapper for merging is:

```text
scripts/run_merge_binary_matrices.py
```

This wrapper:

- loops over chosen datasets
- loops over chosen regimes
- calls `build_binary_matrix.py`
- writes merged response matrices
- saves a merge summary JSON

## Full pipeline summary

The evaluation pipeline is therefore:

1. prepare trained or pretrained model weights
2. evaluate each model on the dataset test split
3. save one `binary_correctness.csv` per model
4. merge those files into one `binary_response_matrix.csv`
5. use the merged matrix as input to IRT

## Example workflows

### Example A: zero-shot evaluation on ImageNet-C

1. run batch evaluation for all models in zero-shot mode
2. save one binary correctness file per model
3. merge them into `results/response_matrices/ImageNet-C/zero_shot/`

### Example B: trained evaluation on CIFAR-100

1. load trained checkpoints for all models
2. evaluate each model on the CIFAR-100 test split
3. save one binary correctness file per model
4. merge them into `results/response_matrices/CIFAR100/trained/`

### Example C: head-only CIFAR-100 evaluation

1. load the saved head-only checkpoints
2. evaluate on the CIFAR-100 test split
3. save one binary correctness file per model
4. merge them into `results/response_matrices/CIFAR100/head_only/`

## Design principles

This part of the repository follows a few important principles:

- evaluation is separated from training
- binary correctness is saved explicitly
- merging is deterministic and transparent
- per-model outputs are preserved before merging
- IRT receives a clean matrix as its input

This design makes the pipeline easier to debug, rerun, and document.

## Notes on alignment

The merge step assumes that all models for a given dataset/regime are evaluated on the same set of items.

To make that alignment robust, the exporter saves:

- `item_id`
- `item_path`
- label metadata

The merge script uses these fields to confirm that all models refer to the same underlying items.

## Recommended execution order

For a typical dataset/regime combination, the recommended order is:

1. `scripts/run_batch_evaluation.py`
2. `scripts/run_merge_binary_matrices.py`
3. IRT fitting scripts in `src/irt/`

## Related files

- `configs/models/models.txt`
- `src/inference/export_binary_correctness.py`
- `scripts/run_batch_evaluation.py`
- `src/inference/build_binary_matrix.py`
- `scripts/run_merge_binary_matrices.py`

## Future extensions

As the repository is refined further, this documentation can be extended with:

- exact example commands for each dataset
- checkpoint naming conventions
- dataset split assumptions for each regime
- IRT script usage examples
- troubleshooting notes for missing or mismatched items
