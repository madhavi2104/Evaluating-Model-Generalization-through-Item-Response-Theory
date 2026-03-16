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
│   └── cifar100/
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

Recommended layout:

```text
data/external/imagenet_sketch/
├── n01440764/
├── n01443537/
└── ...
```

The class ordering and label mapping should be checked carefully so that it matches the ImageNet label space used by the models.

### ImageNet-C

ImageNet-C contains corruption types and severity levels derived from ImageNet validation images.

Recommended layout:

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

- use config files for dataset roots
- avoid absolute personal paths
- document any dataset filtering or subset selection
- document label mappings explicitly
- keep preprocessing deterministic where possible

## What will be documented later

This file is the top-level data guide. As the repository is cleaned, it should later include:

- exact dataset versions
- download sources
- checksum or verification notes where possible
- preprocessing decisions
- subset definitions used in each experiment
- any label remapping required for evaluation

## Summary

Before running the repository, make sure:

1. all required datasets are available locally
2. the directory structure matches the expected format
3. dataset root paths are correctly set in config files
4. any label mapping assumptions are consistent across scripts
