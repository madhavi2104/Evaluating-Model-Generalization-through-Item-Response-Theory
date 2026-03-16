# ============================================================
# save_accuracy_theta_tables.R
# Save accuracy + theta tables from repo-safe outputs
# ============================================================

rm(list = ls())

library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(readr)
library(tools)

# -----------------------------
# Config
# -----------------------------
repo_root <- "."
theta_root <- file.path(repo_root, "results", "analysis", "irt", "fits_best_500")
pred_root  <- file.path(repo_root, "results", "predictions")
out_root   <- file.path(repo_root, "results", "analysis", "irt", "summary_tables")

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

dataset_order <- c("ImageNet", "ImageNet-C", "Sketch", "ImageNet-Sketch", "CIFAR100")
regime_order  <- c("zero_shot", "trained", "head_only")

# -----------------------------
# Helpers
# -----------------------------

find_theta_files <- function(theta_root) {
  files <- list.files(
    theta_root,
    pattern = "^Theta_ModelAbilities_Long\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )

  if (length(files) == 0) {
    stop("No Theta_ModelAbilities_Long.csv files found under: ", theta_root)
  }

  tibble(
    theta_file = files,
    fit_dir = dirname(theta_file),
    fit_label = basename(fit_dir),
    regime = basename(dirname(fit_dir)),
    dataset = basename(dirname(dirname(fit_dir)))
  )
}

find_prediction_files <- function(pred_root) {
  files <- list.files(
    pred_root,
    pattern = "^binary_correctness\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )

  if (length(files) == 0) {
    stop("No binary_correctness.csv files found under: ", pred_root)
  }

  tibble(
    pred_file = files,
    model = basename(dirname(pred_file)),
    regime = basename(dirname(dirname(pred_file))),
    dataset = basename(dirname(dirname(dirname(pred_file))))
  )
}

safe_rank_desc <- function(x) {
  rank(-x, ties.method = "average", na.last = "keep")
}

make_mode_label <- function(dataset, regime) {
  paste(dataset, regime, sep = "_")
}

# -----------------------------
# Theta loading
# -----------------------------

theta_index <- find_theta_files(theta_root)

theta_long <- bind_rows(lapply(seq_len(nrow(theta_index)), function(i) {
  f <- theta_index$theta_file[i]
  dataset <- theta_index$dataset[i]
  regime <- theta_index$regime[i]
  fit_label <- theta_index$fit_label[i]

  df <- fread(f)

  required_cols <- c("Model", "Theta")
  missing_cols <- setdiff(required_cols, names(df))
  if (length(missing_cols) > 0) {
    stop("Missing columns in theta file ", f, ": ", paste(missing_cols, collapse = ", "))
  }

  df %>%
    as.data.frame() %>%
    mutate(
      Dataset = dataset,
      Regime = regime,
      FitLabel = fit_label,
      Mode = make_mode_label(dataset, regime)
    ) %>%
    select(Model, Theta, Dataset, Regime, Mode, FitLabel, everything())
}))

theta_long <- theta_long %>%
  distinct(Model, Dataset, Regime, .keep_all = TRUE) %>%
  mutate(
    Dataset = factor(Dataset, levels = dataset_order),
    Regime = factor(Regime, levels = regime_order)
  ) %>%
  arrange(Dataset, Regime, desc(Theta), Model) %>%
  mutate(
    ThetaRank = ave(
      Theta,
      interaction(Dataset, Regime, drop = TRUE),
      FUN = safe_rank_desc
    )
  ) %>%
  mutate(
    Dataset = as.character(Dataset),
    Regime = as.character(Regime)
  )

# -----------------------------
# Accuracy loading
# -----------------------------

pred_index <- find_prediction_files(pred_root)

accuracy_long <- bind_rows(lapply(seq_len(nrow(pred_index)), function(i) {
  f <- pred_index$pred_file[i]
  dataset <- pred_index$dataset[i]
  regime <- pred_index$regime[i]
  model <- pred_index$model[i]

  df <- fread(f)

  if (!"correct" %in% names(df)) {
    stop("Expected 'correct' column in prediction file: ", f)
  }

  tibble(
    Model = model,
    Dataset = dataset,
    Regime = regime,
    Mode = make_mode_label(dataset, regime),
    Accuracy = mean(df$correct, na.rm = TRUE),
    NItems = nrow(df)
  )
}))

accuracy_long <- accuracy_long %>%
  distinct(Model, Dataset, Regime, .keep_all = TRUE) %>%
  mutate(
    Dataset = factor(Dataset, levels = dataset_order),
    Regime = factor(Regime, levels = regime_order)
  ) %>%
  arrange(Dataset, Regime, desc(Accuracy), Model) %>%
  mutate(
    AccuracyRank = ave(
      Accuracy,
      interaction(Dataset, Regime, drop = TRUE),
      FUN = safe_rank_desc
    )
  ) %>%
  mutate(
    Dataset = as.character(Dataset),
    Regime = as.character(Regime)
  )

# -----------------------------
# Merge theta + accuracy
# -----------------------------

theta_acc_long <- full_join(
  theta_long %>% select(Model, Dataset, Regime, Mode, Theta, ThetaRank, FitLabel),
  accuracy_long %>% select(Model, Dataset, Regime, Mode, Accuracy, AccuracyRank, NItems),
  by = c("Model", "Dataset", "Regime", "Mode")
) %>%
  arrange(factor(Dataset, levels = dataset_order), factor(Regime, levels = regime_order), Model)

# -----------------------------
# Wide tables
# -----------------------------

theta_wide <- theta_long %>%
  select(Model, Mode, Theta) %>%
  pivot_wider(names_from = Mode, values_from = Theta) %>%
  arrange(Model)

theta_rank_wide <- theta_long %>%
  select(Model, Mode, ThetaRank) %>%
  pivot_wider(names_from = Mode, values_from = ThetaRank) %>%
  arrange(Model)

accuracy_wide <- accuracy_long %>%
  select(Model, Mode, Accuracy) %>%
  pivot_wider(names_from = Mode, values_from = Accuracy) %>%
  arrange(Model)

accuracy_rank_wide <- accuracy_long %>%
  select(Model, Mode, AccuracyRank) %>%
  pivot_wider(names_from = Mode, values_from = AccuracyRank) %>%
  arrange(Model)

theta_accuracy_merged_wide <- theta_acc_long %>%
  select(Model, Mode, Theta, Accuracy) %>%
  pivot_wider(
    names_from = Mode,
    values_from = c(Theta, Accuracy),
    names_sep = "__"
  ) %>%
  arrange(Model)

# -----------------------------
# Legacy-style subsets
# -----------------------------
# These are useful because your older downstream scripts often separated:
# 1. zero-shot-only comparisons
# 2. trained/head-only comparisons

theta_long_zero_shot <- theta_long %>%
  filter(Regime == "zero_shot")

accuracy_long_zero_shot <- accuracy_long %>%
  filter(Regime == "zero_shot")

theta_wide_zero_shot <- theta_long_zero_shot %>%
  select(Model, Mode, Theta) %>%
  pivot_wider(names_from = Mode, values_from = Theta) %>%
  arrange(Model)

theta_rank_wide_zero_shot <- theta_long_zero_shot %>%
  select(Model, Mode, ThetaRank) %>%
  pivot_wider(names_from = Mode, values_from = ThetaRank) %>%
  arrange(Model)

accuracy_wide_zero_shot <- accuracy_long_zero_shot %>%
  select(Model, Mode, Accuracy) %>%
  pivot_wider(names_from = Mode, values_from = Accuracy) %>%
  arrange(Model)

accuracy_rank_wide_zero_shot <- accuracy_long_zero_shot %>%
  select(Model, Mode, AccuracyRank) %>%
  pivot_wider(names_from = Mode, values_from = AccuracyRank) %>%
  arrange(Model)

theta_long_trained_like <- theta_long %>%
  filter(Regime %in% c("trained", "head_only"))

accuracy_long_trained_like <- accuracy_long %>%
  filter(Regime %in% c("trained", "head_only"))

theta_wide_trained_like <- theta_long_trained_like %>%
  select(Model, Mode, Theta) %>%
  pivot_wider(names_from = Mode, values_from = Theta) %>%
  arrange(Model)

theta_rank_wide_trained_like <- theta_long_trained_like %>%
  select(Model, Mode, ThetaRank) %>%
  pivot_wider(names_from = Mode, values_from = ThetaRank) %>%
  arrange(Model)

accuracy_wide_trained_like <- accuracy_long_trained_like %>%
  select(Model, Mode, Accuracy) %>%
  pivot_wider(names_from = Mode, values_from = Accuracy) %>%
  arrange(Model)

accuracy_rank_wide_trained_like <- accuracy_long_trained_like %>%
  select(Model, Mode, AccuracyRank) %>%
  pivot_wider(names_from = Mode, values_from = AccuracyRank) %>%
  arrange(Model)

# -----------------------------
# Save outputs
# -----------------------------

write_csv(theta_long, file.path(out_root, "Theta_Long.csv"))
write_csv(theta_wide, file.path(out_root, "Theta_Wide.csv"))
write_csv(theta_rank_wide, file.path(out_root, "Theta_Rank_Wide.csv"))

write_csv(accuracy_long, file.path(out_root, "Accuracy_Long.csv"))
write_csv(accuracy_wide, file.path(out_root, "Accuracy_Wide.csv"))
write_csv(accuracy_rank_wide, file.path(out_root, "Accuracy_Rank_Wide.csv"))

write_csv(theta_acc_long, file.path(out_root, "Theta_Accuracy_Long.csv"))
write_csv(theta_accuracy_merged_wide, file.path(out_root, "Theta_Accuracy_Wide.csv"))

write_csv(theta_long_zero_shot, file.path(out_root, "Theta_Long_ZeroShot.csv"))
write_csv(theta_wide_zero_shot, file.path(out_root, "Theta_Wide_ZeroShot.csv"))
write_csv(theta_rank_wide_zero_shot, file.path(out_root, "Theta_Rank_Wide_ZeroShot.csv"))

write_csv(accuracy_long_zero_shot, file.path(out_root, "Accuracy_Long_ZeroShot.csv"))
write_csv(accuracy_wide_zero_shot, file.path(out_root, "Accuracy_Wide_ZeroShot.csv"))
write_csv(accuracy_rank_wide_zero_shot, file.path(out_root, "Accuracy_Rank_Wide_ZeroShot.csv"))

write_csv(theta_long_trained_like, file.path(out_root, "Theta_Long_TrainedLike.csv"))
write_csv(theta_wide_trained_like, file.path(out_root, "Theta_Wide_TrainedLike.csv"))
write_csv(theta_rank_wide_trained_like, file.path(out_root, "Theta_Rank_Wide_TrainedLike.csv"))

write_csv(accuracy_long_trained_like, file.path(out_root, "Accuracy_Long_TrainedLike.csv"))
write_csv(accuracy_wide_trained_like, file.path(out_root, "Accuracy_Wide_TrainedLike.csv"))
write_csv(accuracy_rank_wide_trained_like, file.path(out_root, "Accuracy_Rank_Wide_TrainedLike.csv"))

# -----------------------------
# Save run metadata
# -----------------------------

metadata <- tibble(
  theta_root = theta_root,
  pred_root = pred_root,
  out_root = out_root,
  n_theta_files = nrow(theta_index),
  n_prediction_files = nrow(pred_index),
  n_theta_rows = nrow(theta_long),
  n_accuracy_rows = nrow(accuracy_long),
  n_joined_rows = nrow(theta_acc_long)
)

write_csv(metadata, file.path(out_root, "Summary_Metadata.csv"))

cat("\nSaved outputs to:\n", out_root, "\n", sep = "")
cat("Done.\n")
