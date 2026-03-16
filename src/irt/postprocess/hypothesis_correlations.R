# === Setup ===
library(dplyr)
library(readr)
library(stringr)
library(ggplot2)
library(reshape2)


# --- Config ---
input_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/analysis"
out_all    <- file.path(input_root, "H1_spearman_results_zeroshot_all.csv")
out_subset <- file.path(input_root, "H1_spearman_results_zeroshot_imagenot6.csv")

# Load rank tables (all models)
acc_rank_all   <- read.csv(file.path(input_root, "all_zeroshot_accuracy_rank_wide.csv"))
theta_rank_all <- read.csv(file.path(input_root, "all_zeroshot_theta_rank_wide.csv"))

stopifnot(all(acc_rank_all$Model == theta_rank_all$Model))

# Define ImageNot model names
imagenot_models <- c("alexnet",
                     "vgg19",
                     "resnet152",
                     "densenet161",
                     "efficientnetv2_l",
                     "convnext_large")

# === Helper: compute correlations ===
compute_correlations <- function(acc_rank, theta_rank) {
  # Ensure numeric
  acc_rank[,-1]   <- lapply(acc_rank[,-1], as.numeric)
  theta_rank[,-1] <- lapply(theta_rank[,-1], as.numeric)
  
  datasets_acc   <- colnames(acc_rank)[-1]
  datasets_theta <- colnames(theta_rank)[-1]
  
  results <- list()
  
  # --- H1.1: Accuracy rank on ImageNet vs accuracy ranks on other datasets ---
  imagenet_acc <- acc_rank[["ImageNet_zeroshot"]]
  for (ds in setdiff(datasets_acc, "ImageNet_zeroshot")) {
    rho <- cor(imagenet_acc, acc_rank[[ds]], method = "spearman", use = "pairwise.complete.obs")
    results[[length(results)+1]] <- data.frame(
      Hypothesis = "H1.1",
      Reference  = "ImageNet_acc",
      Target     = ds,
      SpearmanRho = rho
    )
  }
  
  # --- H1.2: θ vs accuracy ranks on the same dataset ---
  for (ds in datasets_acc) {
    if (ds %in% datasets_theta) {
      rho <- cor(theta_rank[[ds]], acc_rank[[ds]], method = "spearman", use = "pairwise.complete.obs")
      results[[length(results)+1]] <- data.frame(
        Hypothesis = "H1.2",
        Reference  = paste0("Theta_", ds),
        Target     = paste0("Acc_", ds),
        SpearmanRho = rho
      )
    }
  }
  
  # --- H1.3: θ ranks on ImageNet vs θ ranks on other datasets ---
  imagenet_theta <- theta_rank[["ImageNet_zeroshot"]]
  for (ds in setdiff(datasets_theta, "ImageNet_zeroshot")) {
    rho <- cor(imagenet_theta, theta_rank[[ds]], method = "spearman", use = "pairwise.complete.obs")
    results[[length(results)+1]] <- data.frame(
      Hypothesis = "H1.3",
      Reference  = "ImageNet_theta",
      Target     = ds,
      SpearmanRho = rho
    )
  }
  
  # --- H1.4: ImageNet accuracy ranks vs θ ranks on other datasets ---
  for (ds in setdiff(datasets_theta, "ImageNet_zeroshot")) {
    rho <- cor(imagenet_acc, theta_rank[[ds]], method = "spearman", use = "pairwise.complete.obs")
    results[[length(results)+1]] <- data.frame(
      Hypothesis = "H1.4",
      Reference  = "ImageNet_acc",
      Target     = ds,
      SpearmanRho = rho
    )
  }
  
  # --- H1.5: ImageNet θ ranks vs accuracy ranks on other datasets ---
  for (ds in setdiff(datasets_acc, "ImageNet_zeroshot")) {
    rho <- cor(imagenet_theta, acc_rank[[ds]], method = "spearman", use = "pairwise.complete.obs")
    results[[length(results)+1]] <- data.frame(
      Hypothesis = "H1.5",
      Reference  = "ImageNet_theta",
      Target     = ds,
      SpearmanRho = rho
    )
  }
  
  bind_rows(results)
}

# === Run for all models ===
results_all <- compute_correlations(acc_rank_all, theta_rank_all)
write.csv(results_all, out_all, row.names = FALSE)
cat("Saved correlations for ALL models to:", out_all, "\n")

# === Run for 6 ImageNot models only ===
acc_rank_sub   <- acc_rank_all   %>% filter(Model %in% imagenot_models)
theta_rank_sub <- theta_rank_all %>% filter(Model %in% imagenot_models)

results_subset <- compute_correlations(acc_rank_sub, theta_rank_sub)
write.csv(results_subset, out_subset, row.names = FALSE)
cat("Saved correlations for 6 ImageNot models to:", out_subset, "\n")

print(results_all)
print(results_subset)

# =======================
# Correlation Heatmaps
# =======================

# Helper to clean dataset names
clean_names <- function(x) {
  x <- gsub("_zeroshot$", "", x)
  x <- gsub("_trained$", "", x)
  x <- gsub("ImageNet.C", "ImageNet-C", x, fixed = TRUE)
  tools::toTitleCase(x)
}

# Accuracy heatmap
acc_only <- acc_rank_all[,-1]
colnames(acc_only) <- clean_names(colnames(acc_only))
acc_cor <- cor(acc_only, method = "spearman", use = "pairwise.complete.obs")
acc_melt <- reshape2::melt(acc_cor)

p_acc <- ggplot(acc_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.3f", value)), color = "black", size = 3) +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0.5) +
  theme_minimal(base_size = 12) +
  labs(title = "Accuracy Rank Correlations", fill = "Spearman ρ", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(input_root, "accuracy_correlation_heatmap.png"), p_acc, width = 8, height = 6)

# Theta heatmap
theta_only <- theta_rank_all[,-1]
colnames(theta_only) <- clean_names(colnames(theta_only))
theta_cor <- cor(theta_only, method = "spearman", use = "pairwise.complete.obs")
theta_melt <- reshape2::melt(theta_cor)

p_theta <- ggplot(theta_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.3f", value)), color = "black", size = 3) +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0.5) +
  theme_minimal(base_size = 12) +
  labs(title = "Theta Rank Correlations", fill = "Spearman ρ", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(input_root, "theta_correlation_heatmap.png"), p_theta, width = 8, height = 6)

cat("✅ Heatmaps saved with cleaned labels and values:", input_root, "\n")

