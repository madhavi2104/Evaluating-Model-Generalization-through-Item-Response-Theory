# IRT Analysis Pipeline

This document outlines the structure and functionality of the **IRT Analysis Pipeline** within the repository, which is responsible for processing models' predictions, fitting Item Response Theory (IRT) models, and performing advanced statistical analysis.

## Overview 

The **IRT Analysis Pipeline** in this repository is organized into several core modules:

1. Fitting: Scripts dedicated to fitting IRT models (such as 2PL).
2. Postprocess: Scripts that process IRT model outputs and compute key statistics.
3. Diagnostics: Scripts that perform exploratory data analysis and quality checks on the response matrices.

Each submodule in the ```src/irt/``` folder has specific tasks, from data preprocessing and model fitting to post-processing and IRT-based diagnostics.

## Directory Structure

The ``` src/irt/``` directory is organized into the following subfolders:

```text

src/
└── irt/
    ├── fitting/
    │   └── fit_2pl_best_500.R
    ├── postprocess/
    |   ├── save_accuracy_theta_tables.R 
    |   └── hypothesis_correlations.R
    ├── diagnostics/
        ├── matrix_structure_diagnostics.R
        ├── anchor_parameter_stability_by_samplesize.R
        ├── difficulty_alignment.R
        ├── discrimination_alignment.R
        ├── dif_tif_linking.R
        ├── person_fit.R
        ├── multi_dimensionality.R
        └── PCA.R

```

### 1. Fitting 

This folder contains ```fit_2pl_best_500.R``` that fits a 2PL IRT model to a sample size of 500 informative items selected for analysis. It handles the setup, fitting, and saving of the model results.

### 2. Postprocess

The **postprocess** subfolder processes IRT model outputs and computes essential statistics:

* save_accuracy_theta_tables.R: Saves accuracy and ability (θ) estimates and ranks across datasets and models for later analysis.

* hypothesis_correlations.R: Computes correlations between model performance (accuracy) and IRT latent abilities (θ) to test hypotheses about model behavior.

### Diagnostics

The **diagnostics** subfolder contains scripts designed for initial data exploration and quality assurance of response matrices:

* matrix_structure_diagnostics.R: Diagnoses the structure of the response matrix, including missing data and distribution of item difficulties.
* anchor_parameter_stability_by_samplesize.R: Evaluates the stability of item parameters across different sample sizes.
* difficulty_alignment.R: Checks alignment between item difficulties across datasets.
* discrimination_alignment.R: Examines the alignment of item discriminations across datasets.
* dif_tif_linking.R: Analyzes Differential Item Functioning (DIF) and Test Information Functioning (TIF) using IRT models.
* multi_dimensionality.R: Checks for the existence of multidimensionality in the IRT models.
* person_fit.R: Computes person-fit statistics to assess model fit at the individual level.
* PCA.R: Performs PCA on the response matrices to explore the item space and evaluate variance across principal components.

## Detailed Workflow

The typical flow in this pipeline is:

1. **Diagnostics**:
   - **Preprocessing and sample size selection**: Data quality checks and determination of the most stable sample size (```matrix_structure_diagnostics.R``` and ```anchor_parameter_stability_by_samplesize.R```)
   - **Output**: Diagnostics that help ensure the data is suitable for IRT fitting and determining the optimal sample size for model fitting.

2. **Fitting**:
  - **Model fitting**: Using the ```fit_2pl_best_500.R```and sample size determined earlier, the pipeline fits 2PL IRT models to the processed response matrix.
  - **Output**: The fitted models provide latent ability estimates for the models and item parameters.

3. **Postprocessing and Diagnostics**
  - Statistical Analysis: Post-processes the model outputs to compute ability and accuracy rank correlations.
  - Exploratory analysis such as DIF, alignment checks, DIF, PCA and so on.

## Workflow Example

Here is a step-by-step example of running the IRT pipeline:

1. **Preprocessing the data**:
   - Run ```src/irt/diagnostics/matrix_structure_diagnostics.R``` for dataset diagnostics.
   - Run ```src/irt/diagnostics/anchor_parameter_stability_by_samplesize.R``` to determine the optimal sample size for model fitting.

2. **Fitting**:
  - Run ```src/irt/fitting/fit_2pl_best_500.R```with the sample size determined earlier, to fit the 2PL model.

3. **Postprocessing and Diagnostics**
  - Use ```src/irt/postprocess/save_accuracy_theta_tables.R``` and ```src/irt/postprocess/hypothesis_correlations.R``` to analyse the relationship between model performance and latent ability.
  - Run other scripts in ```src/irt/diagnostics/``` to generate IRT-based diagnostics.

## Methodology

The IRT pipeline fits a **2PL (Two-Parameter Logistic)** model to estimate **item difficulty** and **discrimination** parameters. The model assumes that the probability of a correct response depends on:
- **Difficulty**: The threshold of ability required for a model to answer an item correctly.
- **Discrimination**: How well an item distinguishes between models with different abilities.

The IRT model is fitted using the `mirt` package in R, which uses an Expectation-Maximization (EM) algorithm to estimate the model parameters.

## Expected Inputs and Outputs

**Inputs**:
- Binary response matrices (`binary_response_matrix.csv`)
- Model predictions for each dataset and regime

**Outputs**:
- Estimated model ability (`theta`) for each model
- Item difficulty (`b`) and discrimination (`a`) parameters
- Diagnostic plots and tables

## Configuration and Setup

To configure the dataset paths and other settings, modify the `config/config.yml` file. Ensure that the paths to the datasets, model weights, and output directories are correctly set.

Example configuration:
```yaml
paths:
  dataset_root: "/path/to/data/"
  output_dir: "/path/to/output/"
```

## Troubleshooting

**Missing Files**: If the script throws an error about missing files, ensure that the dataset paths are correctly set in the `config/config.yml` file and that all necessary files are downloaded.

**Data Format Issues**: Ensure that the datasets are organized according to the expected folder structure. Refer to `docs/datasets.md` for more details on dataset preparation.

## Notes 

- Reproducibility: All results are saved as CSV files and plots, ensuring the pipeline can be reproduced and analyzed later.
- Future Extensions: As new IRT models or analysis methods are implemented, this pipeline can be extended to incorporate them into the workflow.







