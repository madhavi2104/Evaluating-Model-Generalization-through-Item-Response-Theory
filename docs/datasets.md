# Data Setup and Preparation

This document explains how to set up and prepare the datasets for use in this repository. Since the datasets used in the project are not included due to licensing, size, and distribution restrictions, you must download and prepare them locally before running the pipeline.

## Datasets used

The experiments in this repository use the following datasets:

- **ImageNet**
- **ImageNet-Sketch**
- **ImageNet-C**
- **CIFAR-100**

Due to licensing and size constraints, you need to download these datasets manually. Below, you will find details on the dataset structure, how to download them, and how to set them up for use with the repository.

## General setup principle

All dataset paths should be configured through config files or environment variables. This allows the pipeline to be portable and adaptable to different environments without hardcoding machine-specific absolute paths.

The pipeline is designed so that users can point it to their own local dataset locations by modifying the paths in the configuration file (config/config.yml) or by setting environment variables.

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

### Folder Details:
- `external/`: This directory contains the raw, downloaded datasets. Each dataset should be placed in its respective folder (e.g., imagenet/, cifar100/).
- `processed/`: This directory is for any processed data files. It will be populated with results such as model predictions or preprocessed data during pipeline execution.

Ensure that the internal structure of each dataset follows the standard layout expected by the respective loading code.

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

Recommended raw layout:

```text
data/external/cifar100/
├── train/
│   ├── apple/
│   ├── orange/
│   └── ...
└── val/
    ├── apple/
    ├── orange/
    └── ...
```
## Environment Setup and Configuration

### Configuring Dataset Paths

In the `config/config.yml` file, you can configure the paths for the datasets you have downloaded. The relevant sections should be filled out to point to your local dataset locations.

Example configuration:

``` text
paths:
  imagenet: "/path/to/data/external/imagenet/"
  imagenet_sketch: "/path/to/data/external/imagenet_sketch/"
  imagenet_c: "/path/to/data/external/imagenet_c/"
  cifar100: "/path/to/data/external/cifar100/"
```
Alternatively, you can set environment variables for dataset paths if you prefer not to modify the configuration file directly

``` text
export IMAGE_NET_PATH=/path/to/data/external/imagenet/
export CIFAR100_PATH=/path/to/data/external/cifar100/
```

### Downloading Datasets

For each dataset, follow the respective download link and instructions from the sources. Below are the general steps to download the datasets:

ImageNet: [Link to ImageNet dataset](https://www.image-net.org/)

ImageNet-Sketch: [Link to ImageNet-Sketch dataset](https://github.com/HaohanWang/ImageNet-Sketch)

ImageNet-C: [Link to ImageNet-C dataset](https://github.com/hendrycks/robustness)

CIFAR-100: [Link to CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

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

- `scripts/prepare_data_split.py`

Until then, users should assume that these split folders must already be created locally before training starts.


## Important reproducibility note

To make the repository portable:

- use config files or environment variables for dataset roots
- avoid absolute personal paths
- document any dataset filtering or subset selection
- document label mappings explicitly
- keep preprocessing deterministic where possible
- clearly distinguish between raw dataset layout and training-ready split layout

## Troubleshooting

If you encounter errors related to missing or mismatched files, check the following:
- Ensure that the folder structure matches the expected layout for each dataset.
- Verify that the dataset paths are correctly set in the `config/config.yml` file or environment variables.

## Summary

Before running the repository, make sure:

1. all required datasets are available locally
2. the directory structure matches the expected format
3. dataset root paths are correctly set in config files or environment variables
4. any label mapping assumptions are consistent across scripts
5. required training-ready split folders already exist for training workflows
