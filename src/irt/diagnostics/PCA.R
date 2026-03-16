# --- Load Libraries ---
library(dplyr)
library(readr)
library(ggplot2)
library(viridis)
library(tools)
library(ggpubr)
library(lsa)
library(tibble)
library(pheatmap)

# -----------------------------
# Load configuration
# -----------------------------
config <- yaml.load_file("config/config.yml")

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."

# Choose analysis mode: "zeroshot" OR "trained"
analysis_mode <- "zeroshot"  # change here when needed

datasets <- c("ImageNet", "Sketch", "ImageNet-C", "CIFAR100")
sample_size <- 5000  
pred_root   <- file.path(repo_root, config$paths$predictions)
output_dir <- file.path(repo_root, config$paths$analysis, "Item_PCA", analysis_mode)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# === 2. SAMPLE FUNCTION (stratified per class) ===
sample_items <- function(df, n_total) {
  colnames(df) <- make.names(colnames(df))
  df$True.Label <- trimws(df$True.Label)
  class_count <- length(unique(df$True.Label))
  samples_per_class <- floor(n_total / class_count)
  cat("🔍 Classes:", class_count, "| Samples per class:", samples_per_class, "\n")
  set.seed(42)
  sampled_df <- df %>%
    group_by(True.Label) %>%
    slice_sample(n = samples_per_class) %>%
    ungroup()
  return(sampled_df)
}

# === 3. SAMPLING LOOP FROM DATASETS ===
all_samples <- list()

for (dataset in datasets) {
  cat("🔁 Sampling from:", dataset, "\n")
  
  if (analysis_mode == "zeroshot") {
    # Case 1: all zeroshot
    input_file <- file.path(pred_root,
                            paste0(dataset, "_zeroshot.csv"))
  } else if (analysis_mode == "trained") {
    # Case 2: ImageNet zeroshot, others trained
    if (dataset == "ImageNet") {
      input_file <- file.path(pred_root,
                              paste0(dataset, "_zeroshot.csv"))
    } else {
      input_file <- file.path(pred_root,
                              paste0(dataset, "_trained.csv"))
    }
  } else {
    stop("❌ Unknown analysis_mode: ", analysis_mode)
  }
  
  if (!file.exists(input_file)) {
    warning(paste("❌ File not found:", input_file))
    next
  }
  
  df <- read.csv(input_file)
  colnames(df) <- make.names(colnames(df))
  df$Dataset <- dataset
  sampled_df <- sample_items(df, sample_size)
  all_samples[[dataset]] <- sampled_df
}

combined_df <- bind_rows(all_samples)
cat("✅ Total combined items:", nrow(combined_df), "\n")
print(table(combined_df$Dataset))

# === 4. PCA EXECUTION ===
binary_matrix <- combined_df %>%
  select(-Item, -True.Label, -Dataset)

binary_matrix[] <- lapply(binary_matrix, as.numeric)

# Drop constant / NA-only columns
binary_matrix <- binary_matrix[, apply(binary_matrix, 2, function(x) length(unique(x)) > 1)]
binary_matrix <- binary_matrix[, colSums(is.na(binary_matrix)) == 0]

cat("✅ Binary matrix dimensions before PCA:", dim(binary_matrix), "\n")

pca_res <- prcomp(binary_matrix, center = TRUE, scale. = TRUE)
str(pca_res)
summary(pca_res)


# --- Safely align PCA scores with metadata ---
# After PCA:
item_scores <- as.data.frame(pca_res$x[, 1:2])
item_scores$Dataset <- combined_df$Dataset
item_scores$True.Label <- combined_df$True.Label
item_scores$Item <- combined_df$Item  


# === 5. PLOTS AND COSINE SIMILARITY ===

# PC1 vs PC2 plot
p <- ggplot(item_scores, aes(x = PC1, y = PC2, color = Dataset)) +
  geom_point(alpha = 0.7, size = 2.5) +
  labs(title = "PCA of Combined Items (Item Space)",
       subtitle = paste("Sample size per dataset:", sample_size),
       x = "PC1", y = "PC2") +
  scale_color_viridis_d(option = "D") +
  theme_minimal(base_size = 14)
print(p)
ggsave(filename = file.path(output_dir, "ItemSpace_PC1_PC2.png"),
       plot = p, width = 8, height = 6)

# Faceted PC1 vs PC2 plot by dataset
p <- ggplot(item_scores, aes(x = PC1, y = PC2, color = Dataset)) +
  geom_point(alpha = 0.7, size = 2) +
  facet_wrap(~ Dataset, ncol = 2) +
  labs(title = "PCA of Combined Items (Item Space)",
       subtitle = paste("Sample size per dataset:", sample_size),
       x = "PC1", y = "PC2") +
  scale_color_viridis_d(option = "D", guide = "none") +
  theme_minimal(base_size = 14)
print(p)

# Save to file
ggsave(filename = file.path(output_dir, "ItemSpace_PC1_PC2_Faceted.png"),
       plot = p, width = 10, height = 8)



# Scree Plot (raw eigenvalues)
scree_df <- data.frame(
  PC = 1:length(pca_res$sdev),
  Eigenvalue = pca_res$sdev^2
)
scree_plot_raw <- ggplot(scree_df[1:10, ], aes(x = PC, y = Eigenvalue)) +
  geom_point(size = 2) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 1, color = "red", linetype = "dashed") +
  scale_x_continuous(breaks = 1:10) +
  labs(title = "Scree Plot (Item Space PCA)",
       subtitle = "Red line = Kaiser-Guttman rule (eigenvalue > 1)",
       x = "Principal Component",
       y = "Eigenvalue") +
  theme_minimal(base_size = 14)
print(scree_plot_raw)
# We can understand the item space using just PC1 and PC2.

ggsave(filename = file.path(output_dir, "ItemSpace_ScreePlot_Eigenvalues.png"),
       plot = scree_plot_raw, width = 6, height = 4)

# Save scores
write.csv(item_scores, file.path(output_dir, "ItemSpace_PCA_Scores.csv"), row.names = FALSE)

# Dataset-wise distribution plots
ggplot(item_scores, aes(x = Dataset, y = PC1, fill = Dataset)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.8) +
  labs(title = "PC1 Scores by Dataset (Item Space)", x = "Dataset", y = "PC1 Score") +
  theme_minimal(base_size = 14)
ggplot(item_scores, aes(x = Dataset, y = PC2, fill = Dataset)) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.8) +
  labs(title = "PC2 Scores by Dataset (Item Space)", x = "Dataset", y = "PC2 Score") +
  theme_minimal(base_size = 14)

# Cosine Similarity Matrix from Dataset Means
dataset_means <- item_scores %>%
  group_by(Dataset) %>%
  summarise(mean_PC1 = mean(PC1), mean_PC2 = mean(PC2))
mean_matrix <- dataset_means %>%
  column_to_rownames("Dataset") %>%
  as.matrix()
cos_sim_matrix <- cosine(t(mean_matrix))
rownames(cos_sim_matrix) <- rownames(mean_matrix)
colnames(cos_sim_matrix) <- rownames(mean_matrix)

pheatmap(cos_sim_matrix,
         main = "Cosine Similarity Between Dataset Mean Vectors (PC1 + PC2)",
         display_numbers = TRUE,
         clustering_distance_rows = "euclidean",
         clustering_distance_cols = "euclidean",
         fontsize = 12)

# --- Correlate PCA Loadings with Difficulty per Dataset ---
library(readr)
library(dplyr)
library(ggplot2)
library(viridis)
library(ggpubr)
library(tibble)
library(stringr)

# --- Config ---
irt_root    <- file.path(repo_root, config$paths$theta_estimates)
base_dir <- irt_root
pca_scores_file <- file.path(output_dir, analysis_mode, "ItemSpace_PCA_Scores.csv") 
pca_scores <- read_csv(pca_scores_file)
datasets <- unique(pca_scores$Dataset)

# All outputs go here:
output_root <- dirname(pca_scores_file)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

# --- Helper to process one mode (zeroshot/trained) ---
process_mode <- function(dataset, mode) {
  dataset_dir <- file.path(base_dir, dataset, mode)
  if (!dir.exists(dataset_dir)) {
    warning("❌ No folder for:", dataset, mode)
    return(NULL)
  }
  
  # Find subfolders named best_*
  best_dirs <- list.dirs(dataset_dir, recursive = FALSE, full.names = TRUE)
  best_dirs <- best_dirs[grepl("best_", basename(best_dirs))]
  
  if (length(best_dirs) == 0) {
    warning("⚠️ No best_* folders in:", dataset_dir)
    return(NULL)
  }
  
  for (best_dir in best_dirs) {
    difficulty_path <- file.path(best_dir, "ItemParameters.rds")
    if (!file.exists(difficulty_path)) {
      warning("❌ Missing ItemParameters.rds in:", best_dir)
      next
    }
    
    # Load difficulty & convert rownames → Item column
    difficulty_df <- readRDS(difficulty_path) %>%
      as.data.frame() %>%
      rownames_to_column("Item")
    
    # Match PCA subset
    pca_subset <- pca_scores %>% filter(Dataset == dataset)
    merged_df <- inner_join(pca_subset, difficulty_df, by = "Item")
    cat("✅ Items matched for", dataset, mode, basename(best_dir), ":", nrow(merged_df), "\n")
    
    if (nrow(merged_df) == 0) next
    
    # --- Save merged data ---
    merged_out <- file.path(output_root,
                            paste0(dataset, "_", mode, "_", basename(best_dir), "_PCA_Difficulty_Merged.csv"))
    write.csv(merged_df, merged_out, row.names = FALSE)
    
    # --- Compute correlations ---
    pearson_r_pc1 <- cor(merged_df$PC1, merged_df$d, method = "pearson")
    spearman_r_pc1 <- cor(merged_df$PC1, merged_df$d, method = "spearman")
    pearson_r_pc2 <- cor(merged_df$PC2, merged_df$d, method = "pearson")
    spearman_r_pc2 <- cor(merged_df$PC2, merged_df$d, method = "spearman")
    
    corr_summary <- data.frame(
      Dataset = dataset,
      Mode = mode,
      Sample = basename(best_dir),
      Pearson_PC1 = pearson_r_pc1,
      Spearman_PC1 = spearman_r_pc1,
      Pearson_PC2 = pearson_r_pc2,
      Spearman_PC2 = spearman_r_pc2
    )
    
    # Append correlations to a master CSV
    corr_file <- file.path(output_root, "PCA_Difficulty_Correlations.csv")
    if (!file.exists(corr_file)) {
      write.csv(corr_summary, corr_file, row.names = FALSE)
    } else {
      write.table(corr_summary, corr_file, sep = ",", append = TRUE,
                  col.names = FALSE, row.names = FALSE)
    }
    
    # --- Plot 1: Difficulty vs PC1 ---
    p1 <- ggplot(merged_df, aes(x = PC1, y = d)) +
      geom_point(alpha = 0.6, color = "darkblue") +
      geom_smooth(method = "lm", se = TRUE, color = "red") +
      labs(title = paste0(dataset, " (", mode, " ", basename(best_dir), ")"),
           subtitle = sprintf("PC1 vs d | Pearson r = %.2f, Spearman r = %.2f",
                              pearson_r_pc1, spearman_r_pc1),
           x = "PC1 Score", y = "Item Difficulty (d)") +
      theme_minimal()
    print(p1)
    ggsave(file.path(output_root, paste0(dataset, "_", mode, "_", basename(best_dir), "_PC1_vs_Difficulty.png")),
           p1, width = 6, height = 4)
    
    # --- Plot 2: Difficulty vs PC2 ---
    p2 <- ggplot(merged_df, aes(x = PC2, y = d)) +
      geom_point(alpha = 0.6, color = "darkgreen") +
      geom_smooth(method = "lm", se = TRUE, color = "red") +
      labs(title = paste0(dataset, " (", mode, " ", basename(best_dir), ")"),
           subtitle = sprintf("PC2 vs d | Pearson r = %.2f, Spearman r = %.2f",
                              pearson_r_pc2, spearman_r_pc2),
           x = "PC2 Score", y = "Item Difficulty (d)") +
      theme_minimal()
    print(p2)
    ggsave(file.path(output_root, paste0(dataset, "_", mode, "_", basename(best_dir), "_PC2_vs_Difficulty.png")),
           p2, width = 6, height = 4)
    
    # --- Plot 3: PCA scatter colored by Difficulty ---
    p3 <- ggplot(merged_df, aes(x = PC1, y = PC2, color = d)) +
      geom_point(size = 2, alpha = 0.8) +
      scale_color_viridis_c(option = "C", name = "Difficulty (d)") +
      labs(title = paste0(dataset, " (", mode, " ", basename(best_dir), "): PCA Scatter by Difficulty"),
           x = "PC1", y = "PC2") +
      theme_minimal()
    print(p3)
   ggsave(file.path(output_root, paste0(dataset, "_", mode, "_", basename(best_dir), "_PCA_Scatter_by_Difficulty.png")),
           p3, width = 6.5, height = 5)
  }
}

# =========================
# 1. Analysis: ALL ZEROSHOT
# =========================
cat("\n=== 🔎 Analysis 1: ALL ZEROSHOT ===\n")
for (dataset in datasets) {
  process_mode(dataset, "zeroshot")
}

# ======================================
# 2. Analysis: ImageNet zeroshot vs others trained
# ======================================
cat("\n=== 🔎 Analysis 2: ImageNet Zeroshot vs Others Trained ===\n")

# Step A: ImageNet zeroshot
process_mode("ImageNet", "zeroshot")

# Step B: All others in trained mode
for (dataset in setdiff(datasets, "ImageNet")) {
  process_mode(dataset, "trained")
}

# =====================================================
# === PCA × Discrimination (a) and Difficulty (b) ===
# === FIXED: handle two ANALYSES cleanly (ALL_ZEROSHOT vs MIXED) ===
# === and still load PCA scores per-mode correctly ===
# =====================================================

library(dplyr)
library(ggplot2)
library(viridis)
library(tibble)
library(readr)
library(ggpubr)

# --- CONFIG ---
base_dir <- file.path(repo_root, config$paths$theta_estimates)

# Point to the Item_PCA ROOT (not a specific mode)
pca_root <- file.path(repo_root, config$paths$analysis, "Item_PCA")

modes <- c("zeroshot", "trained")
modes_order <- c("zeroshot", "trained")

# Where to write outputs (put them under Item_PCA/ so it’s consistent)
output_root <- pca_root
dir.create(output_root, showWarnings = FALSE, recursive = TRUE)

output_dir_sideby <- file.path(output_root, "PCA_b_a_sideby")
dir.create(output_dir_sideby, showWarnings = FALSE, recursive = TRUE)

output_dir_vol <- file.path(output_root, "PCA_a_volatility")
dir.create(output_dir_vol, showWarnings = FALSE, recursive = TRUE)

# --- Load PCA scores for BOTH modes, and tag them with Mode ---
read_pca_scores_mode <- function(mode) {
  f <- file.path(pca_root, mode, "ItemSpace_PCA_Scores.csv")
  if (!file.exists(f)) {
    warning(paste0("❌ PCA score file not found for mode=", mode, " at: ", f))
    return(NULL)
  }
  df <- read_csv(f, show_col_types = FALSE)
  
  # Require: Dataset, Item, PC1, PC2
  if (!all(c("Dataset", "Item", "PC1", "PC2") %in% names(df))) {
    stop(paste0(
      "PCA file missing required columns in ", f,
      "\nNeed: Dataset, Item, PC1, PC2\nHave: ",
      paste(names(df), collapse = ", ")
    ))
  }
  
  df$Mode <- mode
  df
}

pca_scores <- bind_rows(lapply(modes, read_pca_scores_mode))
if (nrow(pca_scores) == 0) stop("No PCA scores loaded. Check pca_root / file paths.")

datasets <- sort(unique(pca_scores$Dataset))

# =====================================================
# === IMPORTANT: define TWO ANALYSES (your logic) ===
# 1) ALL_ZEROSHOT: all datasets evaluated in zeroshot
# 2) MIXED: ImageNet zeroshot, all others trained
# =====================================================

jobs <- bind_rows(
  tibble(
    Analysis = "ALL_ZEROSHOT",
    Dataset  = datasets,
    ModeUsed = "zeroshot"
  ),
  bind_rows(
    tibble(Analysis = "MIXED", Dataset = "ImageNet", ModeUsed = "zeroshot"),
    tibble(Analysis = "MIXED", Dataset = setdiff(datasets, "ImageNet"), ModeUsed = "trained")
  )
) %>%
  mutate(
    ModeUsed = factor(ModeUsed, levels = modes_order),
    Analysis = factor(Analysis, levels = c("ALL_ZEROSHOT", "MIXED"))
  )

print(jobs)

# --- Helper: merge PCA with IRT parameters (mode-aware join) ---
merge_pca_with_irt <- function(dataset, mode_used) {
  
  dataset_dir <- file.path(base_dir, dataset, mode_used)
  if (!dir.exists(dataset_dir)) return(NULL)
  
  best_dirs <- list.dirs(dataset_dir, recursive = FALSE, full.names = TRUE)
  best_dirs <- best_dirs[grepl("best_", basename(best_dirs))]
  if (length(best_dirs) == 0) return(NULL)
  
  # Filter PCA scores by BOTH dataset + mode_used
  pca_subset <- pca_scores %>%
    filter(Dataset == dataset, Mode == mode_used)
  
  if (nrow(pca_subset) == 0) return(NULL)
  
  merged_all <- list()
  
  for (best_dir in best_dirs) {
    ip_path <- file.path(best_dir, "ItemParameters.rds")
    if (!file.exists(ip_path)) next
    
    ip <- readRDS(ip_path) %>%
      as.data.frame() %>%
      rownames_to_column("Item")

    # Your code expects: a1 (discrimination), d (intercept)
    if (!all(c("a1", "d") %in% names(ip))) next
    
    # Rename for clarity
    ip <- ip %>%
      dplyr::rename(a = a1, d = d)
    
    # Convert to textbook b:  d = -a*b  =>  b = -d/a
    eps <- 0.2  # you can experiment: 0.1, 0.2, 0.3
    ip <- ip %>%
      dplyr::mutate(
        b = dplyr::if_else(!is.na(a) & abs(a) > eps, -d / a, NA_real_)
      )
    
    merged_df <- inner_join(pca_subset, ip, by = "Item")
    merged_all[[length(merged_all) + 1]] <- merged_df
  }
  
  bind_rows(merged_all)
}

# --- Combine all ANALYSIS × dataset entries (the core fix) ---
merged_all <- bind_rows(lapply(seq_len(nrow(jobs)), function(i) {
  
  ds <- as.character(jobs$Dataset[i])
  md <- as.character(jobs$ModeUsed[i])
  an <- as.character(jobs$Analysis[i])
  
  df <- merge_pca_with_irt(ds, md)
  if (is.null(df) || nrow(df) == 0) return(NULL)
  
  df %>% mutate(Analysis = an, ModeUsed = md)
}))

print(merged_all)

# =====================================================
# === Correlation summary (now grouped by ANALYSIS too) ===
# =====================================================

corr_summary <- merged_all %>%
  group_by(Analysis, Dataset, ModeUsed) %>%
  summarise(
    Spearman_PC1_a = cor(PC1, a, method = "spearman", use = "pairwise.complete.obs"),
    Spearman_PC2_a = cor(PC2, a, method = "spearman", use = "pairwise.complete.obs"),
    Spearman_PC1_b = cor(PC1, b,  method = "spearman", use = "pairwise.complete.obs"),
    Spearman_PC2_b = cor(PC2, b,  method = "spearman", use = "pairwise.complete.obs"),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    ModeUsed = factor(ModeUsed, levels = modes_order),
    Analysis = factor(Analysis, levels = c("ALL_ZEROSHOT", "MIXED"))
  ) %>%
  # For clarity: within each analysis, list zeroshot first then trained
  arrange(Analysis, ModeUsed, Dataset)

write.csv(
  corr_summary,
  file.path(output_root, "PCA_Discrimination_Correlations.csv"),
  row.names = FALSE
)

print(corr_summary)

# =====================================================
# === Visualization: side-by-side b and a (per dataset × analysis) ===
# =====================================================

# Clip outliers for visualization clarity (preserved logic),
# but do it per Dataset × Analysis × ModeUsed so nothing leaks across runs
merged_all <- merged_all %>%
  group_by(Analysis, Dataset, ModeUsed) %>%
  mutate(
    b_clipped = pmin(pmax(d,  quantile(d,  0.01, na.rm = TRUE)),
                     quantile(d,  0.99, na.rm = TRUE)),
    a_clipped = pmin(pmax(a1, quantile(a1, 0.01, na.rm = TRUE)),
                     quantile(a1, 0.99, na.rm = TRUE))
  ) %>%
  ungroup()

# Plot loops: per dataset, per analysis
for (ds in datasets) {
  for (an in levels(jobs$Analysis)) {
    
    df <- merged_all %>% filter(Dataset == ds, Analysis == an)
    if (nrow(df) == 0) next
    
    # You may have just one ModeUsed inside an analysis for a given dataset,
    # but we keep ModeUsed in the title so it's explicit.
    mode_label <- unique(df$ModeUsed)
    mode_label <- paste(mode_label, collapse = ",")
    
    cat("✅ Plotting:", ds, "-", an, "(ModeUsed:", mode_label, ") with", nrow(df), "items\n")
    
    p_b <- ggplot(df, aes(PC1, PC2, color = b_clipped)) +
      geom_point(size = 2, alpha = 0.8) +
      scale_color_viridis_c(option = "C", name = "Difficulty (b)") +
      labs(
        title = paste0(ds, " | ", an),
        subtitle = paste0("ModeUsed: ", mode_label, "  •  PCA item-space colored by difficulty"),
        x = "PC1", y = "PC2"
      ) +
      theme_minimal(base_size = 14)
    
    p_a <- ggplot(df, aes(PC1, PC2, color = a_clipped)) +
      geom_point(size = 2, alpha = 0.8) +
      scale_color_viridis_c(option = "D", name = "Discrimination (a)") +
      labs(
        title = paste0(ds, " | ", an),
        subtitle = paste0("ModeUsed: ", mode_label, "  •  PCA item-space colored by discrimination"),
        x = "PC1", y = "PC2"
      ) +
      theme_minimal(base_size = 14)
    
    combined <- ggarrange(p_b, p_a, ncol = 2, common.legend = FALSE)
    print(combined)
    
    # Uncomment to save
    # ggsave(
    #   file.path(output_dir_sideby, paste0(ds, "_", an, "_PCA_b_a_sideby.png")),
    #   combined, width = 12, height = 5, dpi = 300
    # )
  }
}

cat("\n🎨 All PCA × IRT visualizations ready in:", output_dir_sideby, "\n")
