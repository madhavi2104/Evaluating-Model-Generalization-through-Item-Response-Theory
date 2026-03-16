# === Setup ===
library(dplyr)
library(tidyr)
library(data.table)
library(stringr)

# Paths
theta_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/theta_est"
pred_root  <- "E:/Thesis/IRTNet/output/Hypothesis_2/predictions"
out_root   <- "E:/Thesis/IRTNet/output/Hypothesis_2/analysis"

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# === Helper: load Theta long ===
load_theta <- function(mode_choice) {
  theta_files <- list.files(theta_root,
                            pattern = "Theta_ModelAbilities_Long.csv",
                            recursive = TRUE, full.names = TRUE)
  
  long_list <- lapply(theta_files, function(f) {
    df <- fread(f)
    parts <- str_split(f, .Platform$file.sep)[[1]]
    dataset <- parts[length(parts) - 3]
    mode    <- parts[length(parts) - 2]
    df$Dataset <- dataset
    df$Mode <- mode
    df
  })
  
  theta_long <- bind_rows(long_list)
  
  # Filter according to mode_choice list (dataset -> "trained"/"zeroshot")
  theta_long <- theta_long %>%
    filter(Mode == mode_choice[Dataset])
  
  theta_long
}

# === Helper: load accuracy from prediction files ===
load_accuracy <- function(mode_choice) {
  pred_files <- list.files(pred_root, pattern = "*.csv", recursive = TRUE, full.names = TRUE)
  
  acc_list <- lapply(pred_files, function(f) {
    df <- fread(f)
    # models' predictions are already 0/1 binary after first 2 cols
    model_cols <- df[, -(1:2), with = FALSE]
    
    acc <- colMeans(model_cols, na.rm = TRUE)
    acc_df <- data.frame(Model = names(acc), Accuracy = as.numeric(acc))
    
    parts <- str_split(f, .Platform$file.sep)[[1]]
    filename <- parts[length(parts)]
    dataset <- str_split(filename, "_")[[1]][1]
    mode    <- ifelse(str_detect(filename, "trained"), "trained", "zeroshot")
    
    acc_df$Dataset <- dataset
    acc_df$Mode <- mode
    acc_df
  })
  
  acc_long <- bind_rows(acc_list)
  
  acc_long <- acc_long %>%
    filter(Mode == mode_choice[Dataset])
  
  acc_long
}

# === Helper: make wide + rank tables ===
make_tables <- function(theta_long, acc_long, prefix) {
  # Theta wide
  theta_wide <- theta_long %>%
    mutate(DatasetMode = paste0(Dataset, "_", Mode)) %>%
    select(Model, DatasetMode, Theta) %>%
    pivot_wider(names_from = DatasetMode, values_from = Theta)
  fwrite(theta_wide, file.path(out_root, paste0(prefix, "_theta_wide.csv")))
  
  # Theta rank wide
  theta_rank <- theta_long %>%
    group_by(Dataset, Mode) %>%
    mutate(Rank = rank(-Theta, ties.method = "average")) %>%
    ungroup() %>%
    mutate(DatasetMode = paste0(Dataset, "_", Mode)) %>%
    select(Model, DatasetMode, Rank) %>%
    pivot_wider(names_from = DatasetMode, values_from = Rank)
  fwrite(theta_rank, file.path(out_root, paste0(prefix, "_theta_rank_wide.csv")))
  
  # Accuracy wide
  acc_wide <- acc_long %>%
    mutate(DatasetMode = paste0(Dataset, "_", Mode)) %>%
    select(Model, DatasetMode, Accuracy) %>%
    pivot_wider(names_from = DatasetMode, values_from = Accuracy)
  fwrite(acc_wide, file.path(out_root, paste0(prefix, "_accuracy_wide.csv")))
  
  # Accuracy rank wide
  acc_rank <- acc_long %>%
    group_by(Dataset, Mode) %>%
    mutate(Rank = rank(-Accuracy, ties.method = "average")) %>%
    ungroup() %>%
    mutate(DatasetMode = paste0(Dataset, "_", Mode)) %>%
    select(Model, DatasetMode, Rank) %>%
    pivot_wider(names_from = DatasetMode, values_from = Rank)
  fwrite(acc_rank, file.path(out_root, paste0(prefix, "_accuracy_rank_wide.csv")))
  
  cat("Saved tables for", prefix, "\n")
}

# === Scenario 1: Zeroshot for all datasets ===
datasets <- c("CIFAR100","Sketch","ImageNet-C","ImageNet")
mode_choice1 <- setNames(rep("zeroshot", length(datasets)), datasets)

theta_long1 <- load_theta(mode_choice1)
acc_long1   <- load_accuracy(mode_choice1)
make_tables(theta_long1, acc_long1, "all_zeroshot")

# === Scenario 2: Trained for CIFAR100, Sketch, ImageNet-C; Zeroshot for ImageNet ===
mode_choice2 <- c(CIFAR100 = "trained",
                  Sketch = "trained",
                  `ImageNet-C` = "trained",
                  ImageNet = "zeroshot")

theta_long2 <- load_theta(mode_choice2)
acc_long2   <- load_accuracy(mode_choice2)
make_tables(theta_long2, acc_long2, "mix_trained_zeroshot")

