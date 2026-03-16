# Data Setup

This repository does **not** include the datasets used in the thesis.  
You must download and prepare them locally before running the pipeline.

## Datasets used

The experiments in this repository use the following datasets:

- **ImageNet**
- **ImageNet-Sketch**
- **ImageNet-C**
- **CIFAR-100**

Because of licensing, size, and distribution restrictions, dataset files are not stored in this repository.

## General setup principle

All dataset paths should be configured locally through config files or environment variables.  
Do not hardcode machine-specific absolute paths into the source code.

The repository is designed so that users can point the pipeline to their own local dataset locations.

## Expected local structure

A recommended local structure is:

```text
data/
├── external/
│   ├── imagenet/
│   ├── imagenet_sketch/
│   ├── imagenet_c/
│   ├── cifar100/
│   ├── ImageNet_Sketch_SPLIT/
│   ├── ImageNet-C_split/
│   └── CIFAR100_split/
├── processed/
└── README.md
```

The exact internal structure of each dataset should follow the standard layout expected by the corresponding loading code.

## Dataset notes

### ImageNet

ImageNet should be stored in the standard class-folder format expected by common PyTorch and timm pipelines.

Recommended layout:

```text
data/external/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

If your local copy uses a different validation structure, adapt the data loading config accordingly.

### ImageNet-Sketch

ImageNet-Sketch should be arranged in a class-folder structure compatible with evaluation code.

Recommended raw layout:

```text
data/external/imagenet_sketch/
├── n01440764/
├── n01443537/
└── ...
```

The class ordering and label mapping should be checked carefully so that it matches the ImageNet label space used by the models.

### ImageNet-C

ImageNet-C contains corruption types and severity levels derived from ImageNet validation images.

Recommended raw layout:

```text
data/external/imagenet_c/
├── gaussian_noise/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   └── 5/
├── shot_noise/
├── motion_blur/
└── ...
```

The exact corruption subset, severity levels, and sampling choices used in the thesis should be documented in the experiment configs or analysis notes.

### CIFAR-100

CIFAR-100 may be handled either through automatic dataset download in PyTorch or through a manually prepared local copy, depending on the script.

If stored locally, use a clearly documented path and keep setup consistent across scripts.

## Training-ready dataset splits

Some training scripts in this repository expect datasets to already be arranged into training-ready folder splits.

In particular, the current wrapper scripts expect the following dataset folder names under the chosen data root:

```text
<data-root>/
├── ImageNet_Sketch_SPLIT/
├── ImageNet-C_split/
└── CIFAR100_split/
```

These directories should contain the split structure expected by the training code, typically with subfolders such as:

```text
train/
val/
```

along with the class-folder layout expected by the corresponding backend.

### Important note for training

For this repository, **ImageNet-Sketch** and **ImageNet-C** are not used in their original raw distribution layout for training.  
They must first be reorganized into a training-ready split structure before running the wrapper scripts.

Similarly, if you use the current training wrappers for **CIFAR-100**, they expect a prepared split directory named `CIFAR100_split`.

This means there is an important difference between:

- the **raw dataset form** used for storage or reference
- the **training-ready split form** expected by the training wrappers

## Current wrapper assumption

At the moment, the wrapper scripts assume that the chosen data root already contains the following prepared directories:

- `ImageNet_Sketch_SPLIT`
- `ImageNet-C_split`
- `CIFAR100_split`

If these folders do not already exist, the training wrappers will not work as-is.

## Split creation status

This repository does **not yet automatically generate** these training-ready splits.

If a formal preprocessing or split-generation script is added later, it should be documented in:

- `scripts/01_prepare_data.sh`
- or a dedicated preprocessing document in `docs/`

Until then, users should assume that these split folders must already be created locally before training starts.

## Processed data

The `data/processed/` directory can be used for generated intermediate files such as:

- merged metadata
- label maps
- sampled subsets
- cached preprocessing outputs
- binary response input tables

Raw external datasets should remain separate from processed outputs.

## Important reproducibility note

To make the repository portable:

- use config files or environment variables for dataset roots
- avoid absolute personal paths
- document any dataset filtering or subset selection
- document label mappings explicitly
- keep preprocessing deterministic where possible
- clearly distinguish between raw dataset layout and training-ready split layout

## What will be documented later

This file is the top-level data guide. As the repository is cleaned, it should later include:

- exact dataset versions
- download sources
- checksum or verification notes where possible
- preprocessing decisions
- subset definitions used in each experiment
- any label remapping required for evaluation
- split-generation steps for training-ready datasets

## Summary

Before running the repository, make sure:

1. all required datasets are available locally
2. the directory structure matches the expected format
3. dataset root paths are correctly set in config files or environment variables
4. any label mapping assumptions are consistent across scripts
5. required training-ready split folders already exist for training workflows
