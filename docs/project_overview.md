# Project Overview

This repository contains the reproducible research pipeline for the thesis:

**Evaluating Model Generalization through Item Response Theory**

The project studies how vision models generalize across datasets and how their behavior can be analyzed beyond top-line accuracy using **Item Response Theory (IRT)**.

## Main idea

Standard evaluation often reduces model performance to a single metric such as accuracy. While useful, accuracy does not fully describe how a model behaves across different items, how difficult those items are, or how stable model rankings remain under dataset shift.

This project uses item-level correctness patterns to build a richer analysis pipeline. Instead of asking only whether one model achieves higher accuracy than another, the thesis asks:

- what latent ability is implied by the pattern of correct and incorrect responses
- which items are easy or difficult
- which items best distinguish stronger from weaker models
- how model rankings change across datasets
- how item structure relates to generalization

## Core components of the project

The repository is organized around five major stages:

1. dataset preparation
2. model training or evaluation
3. prediction processing
4. IRT modeling
5. downstream statistical and geometric analysis

## Stage 1: Dataset preparation

The project uses multiple vision datasets with different characteristics:

- **ImageNet**
- **ImageNet-Sketch**
- **ImageNet-C**
- **CIFAR-100**

These datasets are used to compare model behavior under standard evaluation, distribution shift, corruption, and domain transfer.

At this stage, the main tasks are:

- locating the datasets
- validating folder structure
- defining label mappings if needed
- documenting any subset or sampling logic

## Stage 2: Model training or evaluation

Depending on the experiment, models may be used in different ways:

- direct pretrained evaluation
- zero-shot transfer
- head-only adaptation
- full training under a dataset-specific protocol

The repository includes support for models from libraries such as:

- `torchvision`
- `timm`

The goal of this stage is to produce model predictions in a consistent format across datasets and architectures.

## Stage 3: Prediction processing

Model outputs are converted into standardized prediction files and then into **binary correctness responses**.

This step is central because IRT requires a response matrix where each entry represents whether a model answered an item correctly.

Typical outputs from this stage include:

- predicted labels
- true labels
- correctness vectors
- binary response matrices across models and items

This stage creates the bridge between conventional machine learning evaluation and psychometric modeling.

## Stage 4: IRT modeling

The binary response matrices are used to fit IRT models.

The main goals of the IRT stage are to estimate:

- **model ability**
- **item difficulty**
- **item discrimination**

Depending on the final cleaned implementation, this stage may also include:

- model fit diagnostics
- item fit checks
- test characteristic curves
- test information functions
- linking across datasets or settings
- DIF-style comparisons of item behavior

This stage transforms raw correct/incorrect predictions into interpretable latent variables.

## Stage 5: Downstream analysis

Once IRT estimates are available, the repository supports downstream analyses such as:

- rank comparisons between accuracy and latent ability
- rank stability under dataset shift
- cross-dataset correlations
- PCA-based item space analysis
- item geometry and clustering
- figure and table generation for thesis results

These analyses are used to interpret how model behavior changes across domains and whether IRT captures patterns that accuracy alone misses.

## Reproducibility philosophy

This repository is being structured for **staged reproducibility**.

That means users should be able to reproduce the work at different levels:

### Level 1: Final figure reproduction

Generate key figures and tables from saved processed outputs.

### Level 2: Analysis reproduction

Re-run IRT fitting and downstream analyses from saved response matrices or prediction files.

### Level 3: Prediction reproduction

Re-run inference to generate fresh predictions and binary correctness matrices.

### Level 4: Full experiment reproduction

Re-run compute-heavy training or adaptation experiments where resources allow.

This design makes the project usable both for readers who want to inspect the methods and for researchers who want to rerun the full pipeline.

## Repository design principles

The cleaned repository is built around the following principles:

- clear separation between training, inference, IRT, and analysis
- configuration-driven paths and experiment settings
- minimal hardcoded assumptions
- explicit intermediate outputs
- portable directory structure
- support for partial reruns of individual stages

## Planned repository structure

The project is organized into the following conceptual parts:

- `configs/` for dataset, model, and experiment settings
- `data/` for dataset instructions and local setup notes
- `src/training/` for model training logic
- `src/inference/` for prediction and correctness generation
- `src/irt/` for IRT fitting and psychometric analysis
- `src/analysis/` for statistics, PCA, plots, and tables
- `scripts/` for stage-wise execution
- `results/` for generated outputs

## Intended audience

This repository is designed for several kinds of readers:

- thesis examiners who want to inspect the implementation
- researchers interested in model evaluation under dataset shift
- users who want to reproduce selected figures or analyses
- future collaborators or future versions of the project

## Current status

The original thesis codebase is being reorganized into a clean and reproducible structure.  
At this stage, the goal is to preserve the scientific pipeline while improving readability, portability, and rerunnability.

## Next documentation steps

As the repository is refined, this overview should later be complemented by:

- `docs/reproducibility.md`
- `docs/thesis_to_repo_map.md`
- `docs/datasets.md`
- experiment-specific configuration notes
- script-level usage instructions
