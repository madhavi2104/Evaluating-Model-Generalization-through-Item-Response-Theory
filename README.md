# Evaluating Model Generalization through Item Response Theory

This repository is part of my master's thesis, focusing on using Item Response Theory (IRT) to analyze the robustness of machine learning models across various datasets derived from ImageNet. Due to dataset size constraints, this repository does not include the datasets directly but provides instructions on downloading and preparing them for use.

## Overview

Modern vision models are often compared using top-line accuracy, but accuracy alone does not explain **why** models perform differently or how stable their behavior remains across domains. This thesis uses IRT as a framework to study model performance at the level of individual items and to estimate latent model ability from binary correctness patterns.

The repository supports experiments across multiple image datasets and model families, including both:

- **zero-shot / pretrained evaluation**
- **trained or adapted models**, depending on the dataset and setup

The main outputs of the pipeline include:

- prediction files
- binary response matrices
- IRT parameter estimates
- rank comparisons
- correlation analyses
- PCA-based item space analyses
- final figures and tables

## Research Questions

This repository is built around the following broad questions:

- Can IRT provide a more informative measure of model ability than raw accuracy alone?
- How stable are model rankings across dataset shift?
- Which items are consistently easy, hard, or discriminative across datasets?
- How does item structure relate to latent ability and generalization behavior?

## Datasets

The thesis experiments use the following datasets:

- **ImageNet**
- **ImageNet-Sketch**
- **ImageNet-C**
- **CIFAR-100**

These datasets are **not distributed** with this repository.  
You must download them separately and place them in the expected folder structure described in `data/README.md`.

## Models

The project includes experiments with a broad set of pretrained vision architectures, including models from:

- `torchvision`
- `timm`

Depending on the experiment, models are either:

- evaluated directly in a pretrained setting
- trained under a dataset-specific protocol

## What this repository reproduces

This repository is designed to support several levels of reproducibility.

### Level 1: Figure reproduction
Recreate thesis figures and tables from saved processed outputs.

### Level 2: Analysis reproduction
Re-run IRT fitting, rank analysis, PCA analysis, and related statistics from saved prediction or response files.

### Level 3: Prediction reproduction
Re-run model inference to regenerate prediction files and binary correctness matrices.

### Level 4: Full experiment reproduction
Re-run training or adaptation pipelines where compute and dataset access allow.

## Pipeline

The overall workflow is:

1. Prepare datasets
2. Run model training or evaluation
3. Save predictions
4. Convert predictions into binary correctness matrices
5. Fit IRT models
6. Run downstream analyses
7. Generate figures and tables

## Repository Structure

```text
.
├── README.md
├── configs/            # Dataset, model, and experiment configuration files
├── data/               # Data instructions and dataset setup docs
├── docs/               # Additional documentation
├── notebooks/          # Exploration and figure prototyping
├── results/            # Generated outputs, tables, figures, logs
├── scripts/            # High-level run scripts for each stage
└── src/                # Core source code for training, inference, IRT, and analysis

```
## Installation

Clone the repository:

```bash
git clone https://github.com/madhavi2104/Evaluating-Model-Generalization-through-Item-Response-Theory.git
cd Evaluating-Model-Generalization-through-Item-Response-Theory
```

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If the repository is later split into multiple requirement files or a conda environment, those instructions will be added here.

## Data Setup

This repository expects datasets to be stored locally.  
Exact directory layout, dataset notes, and preprocessing assumptions will be documented in:

```text
data/README.md
```

Because some datasets have access restrictions or large storage requirements, this repository only provides the code needed to work with locally prepared copies.

## Running the project

The project is organized as a staged pipeline. A typical run will look like:

```bash
bash scripts/01_prepare_data.sh
bash scripts/02_train.sh
bash scripts/03_predict.sh
bash scripts/04_fit_irt.sh
bash scripts/05_make_figures.sh
```

These scripts are placeholders initially and will be filled in as the repository is cleaned and standardized.

## Outputs

Main outputs produced by the pipeline include:

- **predictions** for each model and dataset
- **binary response matrices** for IRT modeling
- **item parameters** such as difficulty and discrimination
- **ability estimates** for each model
- **correlation and ranking summaries**
- **plots and thesis-ready figures**

Generated outputs will be stored under:

```text
results/
```

## Notes on reproducibility

This project involves multiple datasets, many models, and some compute-heavy stages. Full reproduction may require:

- access to the original datasets
- sufficient storage for predictions and intermediate files
- GPU resources for training-based stages

For practical reproducibility, the repository is being organized so that lighter stages can still be reproduced independently from the most expensive ones.

## Thesis-to-code mapping

A separate document will map thesis sections to code modules and output files:

```text
docs/thesis_to_repo_map.md
```

This is intended to make it easier to connect the written thesis to the implementation.

## Status

This repository is currently being cleaned and reorganized into a fully reproducible structure from the original thesis codebase.

## Citation

If you use this repository, please cite the thesis and repository once citation details are added.

## Contact

For questions about the project, methods, or reproducibility, feel free to open an issue in the repository.
