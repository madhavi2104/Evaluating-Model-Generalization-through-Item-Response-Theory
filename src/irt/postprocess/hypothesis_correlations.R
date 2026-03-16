# ============================================================
# hypothesis_correlations.R
# Repo-safe H1 Spearman correlation tests + heatmaps
# ============================================================

rm(list = ls())

library(dplyr)
library(readr)
library(stringr)
library(ggplot2)
library(reshape2)
library(tools)

# -----------------------------
# Config
# -----------------------------
repo_root <- "."
input_root <- file.path(repo_root, "results", "analysis", "irt", "summary_tables")
out_root   <- file.path(repo_root, "results", "analysis", "irt", "hypothesis_tests")

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

acc_rank_file   <- file.path(input_root, "Accuracy_Rank_Wide_ZeroShot.csv")
theta_rank_file <- file.path(input_root, "Theta_Rank_Wide_ZeroShot.csv")

out_all    <- file.path(out_root, "H1_spearman_results_zeroshot_all.csv")
out_subset <- file.path(out_root, "H1_spearman_results_zeroshot_imagenot6.csv")

imagenot_models <- c(
  "alexnet",
  "vgg19",
  "resnet152",
  "densenet161",
  "efficientnetv2_l",
  "convnext_large"
)

# -----------------------------
# Helpers
# -----------------------------

make_numeric <- function(df) {
  for (col in colnames(df)[-1]) {
    df[[col]] <- as.numeric(df[[col]])
  }
  df
}

clean_names <- function(x) {
  x <- gsub("_zero_shot$", "", x)
  x <- gsub("_zeroshot$", "", x)
  x <- gsub("_trained$", "", x)
  x <- gsub("_head_only$", "", x)
  x <- gsub("ImageNet\\.C", "ImageNet-C", x)
  x <- gsub("Imagenet Sketch", "ImageNet-Sketch", tools::toTitleCase(x), fixed = TRUE)
  x
}

compute_correlations <- function(acc_rank, theta_rank) {
  acc_rank <- make_numeric(acc_rank)
  theta_rank <- make_numeric(theta_rank)

  stopifnot(all(acc_rank$Model == theta_rank$Model))

  datasets_acc   <- colnames(acc_rank)[-1]
  datasets_theta <- colnames(theta_rank)[-1]

  if (!"ImageNet_zero_shot" %in% datasets_acc && "ImageNet_zeroshot" %in% datasets_acc) {
    names(acc_rank)[names(acc_rank) == "ImageNet_zeroshot"] <- "ImageNet_zero_shot"
    datasets_acc <- colnames(acc_rank)[-1]
  }
  if (!"ImageNet_zero_shot" %in% datasets_theta && "ImageNet_zeroshot" %in% datasets_theta) {
    names(theta_rank)[names(theta_rank) == "ImageNet_zeroshot"] <- "ImageNet_zero_shot"
    datasets_theta <- colnames(theta_rank)[-1]
  }

  if (!"ImageNet_zero_shot" %in% datasets_acc || !"ImageNet_zero_shot" %in% datasets_theta) {
    stop("Expected ImageNet zero-shot column in both accuracy and theta rank tables.")
  }

  results <- list()

  imagenet_acc   <- acc_rank[["ImageNet_zero_shot"]]
  imagenet_theta <- theta_rank[["ImageNet_zero_shot"]]

  # --- H1.1: ImageNet accuracy rank vs accuracy ranks on other datasets
  for (ds in setdiff(datasets_acc, "ImageNet_zero_shot")) {
    rho <- cor(
      imagenet_acc, acc_rank[[ds]],
      method = "spearman",
      use = "pairwise.complete.obs"
    )
    results[[length(results) + 1]] <- data.frame(
      Hypothesis = "H1.1",
      Reference = "ImageNet_acc",
      Target = ds,
      SpearmanRho = rho,
      stringsAsFactors = FALSE
    )
  }

  # --- H1.2: theta vs accuracy ranks on same dataset
  for (ds in intersect(datasets_acc, datasets_theta)) {
    rho <- cor(
      theta_rank[[ds]], acc_rank[[ds]],
      method = "spearman",
      use = "pairwise.complete.obs"
    )
    results[[length(results) + 1]] <- data.frame(
      Hypothesis = "H1.2",
      Reference = paste0("Theta_", ds),
      Target = paste0("Acc_", ds),
      SpearmanRho = rho,
      stringsAsFactors = FALSE
    )
  }

  # --- H1.3: ImageNet theta rank vs theta ranks on other datasets
  for (ds in setdiff(datasets_theta, "ImageNet_zero_shot")) {
    rho <- cor(
      imagenet_theta, theta_rank[[ds]],
      method = "spearman",
      use = "pairwise.complete.obs"
    )
    results[[length(results) + 1]] <- data.frame(
      Hypothesis = "H1.3",
      Reference = "ImageNet_theta",
      Target = ds,
      SpearmanRho = rho,
      stringsAsFactors = FALSE
    )
  }

  # --- H1.4: ImageNet accuracy rank vs theta ranks on other datasets
  for (ds in setdiff(datasets_theta, "ImageNet_zero_shot")) {
    rho <- cor(
      imagenet_acc, theta_rank[[ds]],
      method = "spearman",
      use = "pairwise.complete.obs"
    )
    results[[length(results) + 1]] <- data.frame(
      Hypothesis = "H1.4",
      Reference = "ImageNet_acc",
      Target = ds,
      SpearmanRho = rho,
      stringsAsFactors = FALSE
    )
  }

  # --- H1.5: ImageNet theta rank vs accuracy ranks on other datasets
  for (ds in setdiff(datasets_acc, "ImageNet_zero_shot")) {
    rho <- cor(
      imagenet_theta, acc_rank[[ds]],
      method = "spearman",
      use = "pairwise.complete.obs"
    )
    results[[length(results) + 1]] <- data.frame(
      Hypothesis = "H1.5",
      Reference = "ImageNet_theta",
      Target = ds,
      SpearmanRho = rho,
      stringsAsFactors = FALSE
    )
  }

  bind_rows(results)
}

save_heatmap <- function(df_wide, title_text, out_file) {
  mat_df <- df_wide[, -1, drop = FALSE]
  mat_df <- make_numeric(mat_df)
  colnames(mat_df) <- clean_names(colnames(mat_df))

  cor_mat <- cor(mat_df, method = "spearman", use = "pairwise.complete.obs")
  cor_melt <- reshape2::melt(cor_mat)

  p <- ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.3f", value)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "red",
      mid = "white",
      high = "blue",
      midpoint = 0.5,
      limits = c(-1, 1)
    ) +
    theme_minimal(base_size = 12) +
    labs(
      title = title_text,
      fill = "Spearman rho",
      x = "",
      y = ""
    ) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid = element_blank()
    )

  ggsave(out_file, p, width = 8, height = 6, dpi = 300)
}

# -----------------------------
# Load rank tables
# -----------------------------

if (!file.exists(acc_rank_file)) {
  stop("Missing file: ", acc_rank_file)
}
if (!file.exists(theta_rank_file)) {
  stop("Missing file: ", theta_rank_file)
}

acc_rank_all   <- read.csv(acc_rank_file, stringsAsFactors = FALSE)
theta_rank_all <- read.csv(theta_rank_file, stringsAsFactors = FALSE)

stopifnot(all(acc_rank_all$Model == theta_rank_all$Model))

# -----------------------------
# Run H1 correlations
# -----------------------------

results_all <- compute_correlations(acc_rank_all, theta_rank_all)
write.csv(results_all, out_all, row.names = FALSE)

acc_rank_sub   <- acc_rank_all   %>% filter(tolower(Model) %in% imagenot_models)
theta_rank_sub <- theta_rank_all %>% filter(tolower(Model) %in% imagenot_models)

stopifnot(all(acc_rank_sub$Model == theta_rank_sub$Model))

results_subset <- compute_correlations(acc_rank_sub, theta_rank_sub)
write.csv(results_subset, out_subset, row.names = FALSE)

print(results_all)
print(results_subset)

# -----------------------------
# Heatmaps
# -----------------------------

save_heatmap(
  acc_rank_all,
  title_text = "Accuracy Rank Correlations (Zero-Shot)",
  out_file = file.path(out_root, "accuracy_correlation_heatmap_zeroshot.png")
)

save_heatmap(
  theta_rank_all,
  title_text = "Theta Rank Correlations (Zero-Shot)",
  out_file = file.path(out_root, "theta_correlation_heatmap_zeroshot.png")
)

# -----------------------------
# Metadata / quick summary
# -----------------------------

summary_df <- data.frame(
  Output = c(
    "H1_spearman_results_zeroshot_all.csv",
    "H1_spearman_results_zeroshot_imagenot6.csv",
    "accuracy_correlation_heatmap_zeroshot.png",
    "theta_correlation_heatmap_zeroshot.png"
  ),
  Path = c(
    out_all,
    out_subset,
    file.path(out_root, "accuracy_correlation_heatmap_zeroshot.png"),
    file.path(out_root, "theta_correlation_heatmap_zeroshot.png")
  ),
  stringsAsFactors = FALSE
)

write.csv(summary_df, file.path(out_root, "Hypothesis_Test_Outputs.csv"), row.names = FALSE)

cat("Saved all-model H1 correlations to: ", out_all, "\n", sep = "")
cat("Saved ImageNot-6 H1 correlations to: ", out_subset, "\n", sep = "")
cat("Saved zero-shot heatmaps to: ", out_root, "\n", sep = "")
cat("Done.\n")
