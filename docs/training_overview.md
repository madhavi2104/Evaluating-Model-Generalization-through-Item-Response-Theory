# Training Overview

This document describes the training-related code in the repository and how it is organized.

## Overview

The thesis codebase uses more than one training pipeline. This is because different groups of models were trained or adapted using different software stacks and protocols.

The repository currently includes three training routes:

1. a **timm-based training backend**
2. a **torchvision-based training backend**
3. a **head-only CIFAR-100 adaptation pipeline**

These are kept separate in the cleaned repository because they represent genuinely different workflows.

## 1. TIMM backend

The timm backend is used for training or fine-tuning models implemented through the `timm` library.

### Main files

- `src/training/backends/timm/train.py`
- `scripts/train_timm_wrapper.py`

### Responsibilities

The timm backend handles:

- model creation through `timm`
- training and evaluation loops
- checkpoint saving
- optimizer and scheduler setup
- experiment execution for timm-compatible architectures

The wrapper script is used to launch training runs with dataset-specific paths, output paths, and model-specific settings.

## 2. Torchvision backend

The torchvision backend is a more self-contained training stack with helper modules for transforms, samplers, and utilities.

### Main files

- `src/training/backends/torchvision/train.py`
- `src/training/backends/torchvision/presets.py`
- `src/training/backends/torchvision/transforms.py`
- `src/training/backends/torchvision/sampler.py`
- `src/training/backends/torchvision/utils.py`
- `scripts/train_torchvision_wrapper.py`

### Responsibilities

The torchvision backend handles:

- model creation through `torchvision`
- training and validation loops
- augmentation presets
- repeated augmentation sampling
- logging and metric tracking
- checkpoint and resume logic
- experiment execution for torchvision-compatible architectures

The helper modules remain grouped with this backend because they are part of the same training implementation.

## 3. Head-only CIFAR-100 adaptation

The repository also includes a separate script for head-only adaptation on CIFAR-100.

### Main file

- `src/training/protocols/head_only_cifar100.py`

### Responsibilities

This script is a dedicated protocol rather than a generic backend. It is used to:

- load ImageNet-pretrained vision models
- replace the original classifier head with a CIFAR-100 head
- freeze the pretrained backbone
- train only the classifier head on CIFAR-100
- evaluate the adapted model on the CIFAR-100 test split
- export binary correctness outputs for downstream IRT analysis

This workflow is kept separate because it is conceptually different from the main full-training pipelines.

### Important clarification

Although this protocol starts from ImageNet-pretrained weights, it is **not strict zero-shot evaluation** on CIFAR-100.  
It is a **head-only supervised adaptation protocol** using pretrained initialization.

That distinction matters for reproducibility and for matching the experimental setup described in the thesis.

## Wrappers vs core training files

A key distinction in this repository is the difference between:

- **core training files**, which contain the actual training logic
- **wrapper scripts**, which launch runs with chosen paths and settings

Wrapper scripts should stay thin. Over time, machine-specific hardcoded paths should be removed from them and replaced with config-driven inputs.

## How to launch training

The wrapper scripts are used to launch backend-specific training runs.

### TIMM example

```bash
python scripts/train_timm_wrapper.py convnext_base CIFAR100 \
  --data-root /path/to/data \
  --output-root /path/to/results
```

### Torchvision example

```bash
python scripts/train_torchvision_wrapper.py resnet50 Sketch \
  --data-root /path/to/data \
  --output-root /path/to/results
```

### Head-only CIFAR-100 example

```bash
python src/training/protocols/head_only_cifar100.py \
  --model resnet50 \
  --data-root /path/to/CIFAR100_split \
  --output-root /path/to/results \
  --epochs 10
```

## Environment variable option

Instead of passing paths every time, you can set:

```bash
export THESIS_DATA_ROOT=/path/to/data
export THESIS_OUTPUT_ROOT=/path/to/results
```

and then run:

```bash
python scripts/train_timm_wrapper.py convnext_base CIFAR100
python scripts/train_torchvision_wrapper.py resnet50 Sketch
python src/training/protocols/head_only_cifar100.py --model resnet50
```

## Dataset assumption

These wrappers expect training-ready dataset directories to already exist under the chosen data root.

The current setup expects split folders such as:

- `ImageNet_Sketch_SPLIT`
- `ImageNet-C_split`
- `CIFAR100_split`

These dataset preparation steps must be completed before training is launched.

For the head-only CIFAR-100 protocol, the script expects a prepared directory with:

- `train/`
- `test/`

under the selected CIFAR-100 root.

## What the wrappers do

The wrapper scripts are responsible for:

- selecting the correct dataset directory
- selecting the output directory
- resolving the backend training script
- applying model-specific batch size or worker overrides
- launching the actual training script with a consistent command

They are **not** intended to perform dataset splitting or preprocessing themselves.

## Planned cleanup

The main cleanup tasks for the training code are:

- remove hardcoded absolute paths
- rename helper files to remove `_local`
- standardize output directory naming
- unify logging conventions across backends
- move dataset and output paths into config files
- document expected inputs and outputs for each training route

## Design principle

The cleaned repository does not try to force all training code into one artificial script. Instead, it preserves the real structure of the project while making it clearer, more portable, and easier to rerun.
