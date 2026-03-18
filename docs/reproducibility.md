# Reproducibility

This document outlines how to reproduce the results in this repository
at different levels. The goal is to provide users with clear
instructions on how to replicate the experiments, generate the same
results, and ensure that the pipeline is fully reproducible.

The repository is structured to allow **staged reproducibility**,
meaning you can reproduce the work at different levels:

-   **Level 1: Analysis reproduction**

-   **Level 2: Prediction reproduction**

-   **Level 3: Full experiment reproduction**


## Level 1: Analysis Reproduction

This level allows you to reproduce the analysis (e.g., IRT fitting and
statistical analysis) using the saved response matrices or prediction
files.

**Instructions:**

1.  **Download Pre-processed Data**:

    -   Download the **response matrices** and **model predictions**
        from the `results/` directory.

2.  **Re-run IRT Fitting**:

    -   Use the following command to re-run IRT fitting on the saved
        binary response matrices:

    <!-- -->

            Rscript src/irt/fitting/fit_2pl_best_500.R --input-dir <path_to_response_matrices>

3.  **Postprocessing**:

    -   After fitting the IRT model, you can run the **postprocessing**
        scripts to compute the key statistics:

    <!-- -->

            Rscript src/irt/postprocess/save_accuracy_theta_tables.R --input-dir <path_to_fitted_model_results>

4.  **Results**:

    -   This will re-generate the statistical outputs used in the
        thesis, such as model ability estimates, item difficulty, and
        discrimination parameters.

## Level 2: Prediction Reproduction

At this level, users can re-run inference to generate **fresh
predictions** and **binary correctness matrices**.

**Instructions:**

1.  **Configure Dataset Paths**:

    -   Ensure that the dataset paths are correctly set in the
        `config/config.yml` file or as environment variables. Example:

    <!-- -->

            paths:
              imagenet: "/path/to/data/external/imagenet/"
              cifar100: "/path/to/data/external/cifar100/"

2.  **Re-run Inference**:

    -   Use the following script to re-run inference for each model:

    <!-- -->

            python src/inference/export_binary_correctness.py --model <model_name> --dataset ImageNet --regime zeroshot --data-root /path/to/data

3.  **Merge Outputs**:

    -   After running inference for all models, use the following script
        to merge the per-model outputs into one binary response matrix:

    <!-- -->

            python scripts/run_merge_binary_matrices.py --data-root /path/to/data

4.  **Results**:

    -   This will output the merged **binary response matrix**, which
        can be used for IRT analysis in subsequent steps.

## Level 3: Full Experiment Reproduction

This is the most advanced level and involves **re-running compute-heavy
training or adaptation experiments**.

**Instructions:**

1.  **Configure Training Settings**:

    -   Set the paths for your datasets and model checkpoints in the
        `config/config.yml` file.

2.  **Run Training**:

    -   Use the wrapper scripts for each backend to train models:

    <!-- -->

    -   **TIMM**:

                    python scripts/train_timm_wrapper.py --model <model_name> --dataset ImageNet --data-root /path/to/data

    -   **Torchvision**:

                    python scripts/train_torchvision_wrapper.py --model <model_name> --dataset CIFAR-100 --data-root /path/to/data

3.  **Save Model Checkpoints**:

    -   Training will save model checkpoints in the
        `results/checkpoints/` directory.

4.  **Run Evaluation**:

    -   After training, evaluate the models using
        `scripts/run_batch_evaluation.py` to generate prediction files.

5.  **Run Full Pipeline**:

    -   Use the scripts to merge the binary response matrices and proceed with IRT modeling as outlined in Level 3.

## Final Notes

-   Ensure that all **dependencies** are installed, and the environment
    is set up as per the instructions in `docs/setup.md`.

-   For any specific **dataset or model issues**, refer to the
    `docs/datasets.md` for detailed setup instructions.

-   **Troubleshooting** steps are provided in the
    `docs/troubleshooting.md` for common errors during training,
    evaluation, or IRT analysis.
