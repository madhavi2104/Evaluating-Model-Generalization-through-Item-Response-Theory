# ============================================================
# matrix_structure_diagnostics.R
# Investigate response matrix structure BEFORE sampling
# (Updated: adds p(correct) vs item index ordering diagnostics)
# (Updated v2: formal near-deterministic definitions via thresholds)
# ============================================================

rm(list = ls())

library(dplyr)
library(data.table)
library(ggplot2)
library(stringr)
library(tools)

# -----------------------------
# Config
# -----------------------------
pred_dir <- "E:/Thesis/IRTNet/output/Hypothesis_2/predictions"

test_files <- c(
  "ImageNet_zeroshot.csv",
  "ImageNet-C_zeroshot.csv",
  "ImageNet-C_trained.csv",
  "Sketch_zeroshot.csv",
  "Sketch_trained.csv",
  "CIFAR100_zeroshot.csv",
  "CIFAR100_trained.csv"
)

out_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/analysis/IRT_based/matrix_structure_diagnostics"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# Correlation / PCA limits for speed
max_items_for_pca  <- 5000
corr_item_sample   <- 800
corr_model_min     <- 20

# NEW: near-deterministic thresholds (make them explicit for referencing in the thesis)
eps_var <- 1e-6     # item is "near-deterministic" if Var(x_.i) < eps_var
p_lo <- 0.05        # item is "very hard" if p(correct) < p_lo
p_hi <- 0.95        # item is "very easy" if p(correct) > p_hi

# Ordering diagnostics params
rolling_window <- 200   # rolling mean window for p(correct) trend
bin_size <- 200         # bin size for binned p(correct) vs index

# -----------------------------
# Helpers
# -----------------------------

read_prediction_csv <- function(path) {
  df <- fread(path)
  
  meta_cols <- intersect(names(df), c("Item", "Class", "Label", "Synset", "Superclass", "Dataset"))
  non_num_cols <- names(df)[!sapply(df, is.numeric)]
  meta_cols <- unique(c(meta_cols, non_num_cols))
  
  cand <- setdiff(names(df), meta_cols)
  
  X <- df[, ..cand]
  X <- as.data.frame(lapply(X, function(x) as.numeric(as.character(x))))
  
  keep_rows <- complete.cases(X)
  X <- X[keep_rows, , drop = FALSE]
  df <- df[keep_rows, ]
  
  # models x items
  X_mat <- t(as.matrix(X))
  
  item_id <- if ("Item" %in% names(df)) df$Item else paste0("item_", seq_len(nrow(df)))
  
  group <- NA_character_
  for (g in c("Class", "Synset", "Superclass", "Label")) {
    if (g %in% names(df)) {
      group <- as.character(df[[g]])
      break
    }
  }
  if (length(group) == 1 && is.na(group)) group <- rep(NA_character_, length(item_id))
  
  colnames(X_mat) <- item_id
  rownames(X_mat) <- colnames(X)
  
  list(
    X = X_mat,      # models x items
    item_id = item_id,
    group = group,
    raw = df
  )
}

compute_running_group_coverage <- function(group_vec, window = 200) {
  if (all(is.na(group_vec))) return(NULL)
  
  n <- length(group_vec)
  if (n < window) window <- max(10, floor(n/5))
  
  cum_unique <- sapply(seq_len(n), function(i) length(unique(group_vec[1:i])))
  win_unique <- sapply(seq_len(n), function(i) {
    lo <- max(1, i - window + 1)
    length(unique(group_vec[lo:i]))
  })
  
  data.frame(
    idx = seq_len(n),
    cum_unique_groups = cum_unique,
    win_unique_groups = win_unique,
    window = window
  )
}

summarise_items <- function(X) {
  p <- colMeans(X)
  v <- apply(X, 2, var)
  data.frame(
    Item = colnames(X),
    p_correct = as.numeric(p),
    var = as.numeric(v)
  )
}

estimate_item_corr_stats <- function(X, max_items = 800) {
  M <- nrow(X); I <- ncol(X)
  if (M < corr_model_min || I < 5) return(NULL)
  
  set.seed(1)
  items <- colnames(X)
  if (length(items) > max_items) items <- sample(items, max_items)
  
  Xs <- X[, items, drop = FALSE]
  C <- suppressWarnings(cor(Xs, use = "pairwise.complete.obs"))
  
  off <- C[upper.tri(C)]
  off <- off[is.finite(off)]
  
  data.frame(
    n_items_used = length(items),
    corr_median = median(off),
    corr_q95 = quantile(off, 0.95),
    corr_q99 = quantile(off, 0.99),
    frac_gt_0.9 = mean(off > 0.9),
    frac_gt_0.95 = mean(off > 0.95)
  )
}

item_pca <- function(X, max_items = 5000) {
  I <- ncol(X)
  if (I < 10) return(NULL)
  
  set.seed(1)
  items <- colnames(X)
  if (length(items) > max_items) items <- sample(items, max_items)
  
  Y <- t(X[, items, drop = FALSE])  # items x models
  Y <- scale(Y, center = TRUE, scale = FALSE)
  
  pca <- prcomp(Y, rank. = 10)
  list(
    items = items,
    scores = as.data.frame(pca$x),
    sdev = pca$sdev
  )
}

rolling_mean <- function(x, k) {
  if (length(x) < k) return(rep(mean(x), length(x)))
  stats::filter(x, rep(1 / k, k), sides = 1) |> as.numeric()
}

bin_index_summary <- function(p, bin_size) {
  n <- length(p)
  idx <- seq_len(n)
  bin <- ((idx - 1) %/% bin_size) + 1
  data.frame(idx = idx, p = p, bin = bin) %>%
    group_by(bin) %>%
    summarise(
      idx_mid = round(mean(idx)),
      p_mean = mean(p),
      p_sd = sd(p),
      .groups = "drop"
    )
}

# -----------------------------
# Main
# -----------------------------
all_summaries <- list()

for (f in test_files) {
  message("\n============================")
  message("Processing: ", f)
  
  path <- file.path(pred_dir, f)
  base <- file_path_sans_ext(basename(f))
  
  parts <- strsplit(base, "_")[[1]]
  dataset <- parts[1]
  mode <- ifelse(str_detect(base, "trained"), "trained", "zeroshot")
  
  out_dir <- file.path(out_root, dataset, mode)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  dat <- read_prediction_csv(path)
  X <- dat$X
  group <- dat$group
  
  message("Matrix dims (models x items): ", nrow(X), " x ", ncol(X))
  
  # --- 1) Implicit ordering check: group coverage ---
  run_cov <- compute_running_group_coverage(group, window = 200)
  if (!is.null(run_cov)) {
    p_cov <- ggplot(run_cov, aes(idx, cum_unique_groups)) +
      geom_line() +
      labs(
        title = paste0(base, ": cumulative unique groups over item order"),
        x = "Item index (original order)",
        y = "Cumulative unique groups"
      ) +
      theme_minimal()
    
    ggsave(file.path(out_dir, "ordering_cum_group_coverage.pdf"), p_cov, width = 8, height = 4)
    
    p_win <- ggplot(run_cov, aes(idx, win_unique_groups)) +
      geom_line() +
      labs(
        title = paste0(base, ": unique groups in rolling window"),
        subtitle = paste0("Window size = ", unique(run_cov$window)),
        x = "Item index (original order)",
        y = "Unique groups (rolling window)"
      ) +
      theme_minimal()
    
    ggsave(file.path(out_dir, "ordering_window_group_coverage.pdf"), p_win, width = 8, height = 4)
  }
  
  # --- 2) Per-item variation ---
  item_df <- summarise_items(X)
  item_df$idx <- seq_len(nrow(item_df))
  
  # NEW: tag near-deterministic items using eps_var for later reference
  item_df$is_near_det <- item_df$var < eps_var
  
  fwrite(item_df, file.path(out_dir, "item_variation.csv"))
  
  p_hist_p <- ggplot(item_df, aes(p_correct)) + geom_histogram(bins = 50) +
    geom_vline(xintercept = c(p_lo, 0.5, p_hi), linetype = "dashed") +
    labs(title = paste0(base, ": item mean correctness"),
         x = "p(correct) across models", y = "count") +
    theme_minimal()
  ggsave(file.path(out_dir, "hist_item_pcorrect.pdf"), p_hist_p, width = 7, height = 4)
  
  p_hist_v <- ggplot(item_df, aes(var)) + geom_histogram(bins = 50) +
    labs(title = paste0(base, ": item variance"),
         x = "var across models", y = "count") +
    theme_minimal()
  ggsave(file.path(out_dir, "hist_item_var.pdf"), p_hist_v, width = 7, height = 4)
  
  # --- 2b) Difficulty ordering: p(correct) vs item index ---
  k <- min(rolling_window, nrow(item_df))
  item_df$p_roll <- rolling_mean(item_df$p_correct, k)
  
  rho <- suppressWarnings(cor(item_df$idx, item_df$p_correct, method = "spearman"))
  fwrite(
    data.frame(File = f, Dataset = dataset, Mode = mode,
               spearman_idx_p = rho, rolling_window = k),
    file.path(out_dir, "ordering_index_pcorrect_corr.csv")
  )
  
  p_order <- ggplot(item_df, aes(idx, p_correct)) +
    geom_point(alpha = 0.25, size = 0.7) +
    geom_line(aes(y = p_roll), linewidth = 0.7) +
    labs(
      title = paste0(base, ": p(correct) vs item index"),
      subtitle = paste0("Spearman(idx, p) = ", round(rho, 3),
                        " | rolling window = ", k),
      x = "Item index (original order)",
      y = "p(correct) across models"
    ) +
    theme_minimal()
  ggsave(file.path(out_dir, "ordering_pcorrect_vs_index.pdf"), p_order, width = 8, height = 4.5)
  
  bdf <- bin_index_summary(item_df$p_correct, bin_size = min(bin_size, nrow(item_df)))
  p_bins <- ggplot(bdf, aes(idx_mid, p_mean)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.4) +
    geom_ribbon(aes(ymin = p_mean - p_sd, ymax = p_mean + p_sd), alpha = 0.2) +
    labs(
      title = paste0(base, ": binned p(correct) vs item index"),
      subtitle = paste0("bin_size = ", min(bin_size, nrow(item_df)),
                        " | ribbon = ±1 SD within bin"),
      x = "Item index (bin midpoints)",
      y = "Mean p(correct) in bin"
    ) +
    theme_minimal()
  ggsave(file.path(out_dir, "ordering_pcorrect_vs_index_binned.pdf"), p_bins, width = 8, height = 4.5)
  
  # --- 3) Column similarity / redundancy ---
  corr_stats <- estimate_item_corr_stats(X, max_items = corr_item_sample)
  if (!is.null(corr_stats)) {
    fwrite(corr_stats, file.path(out_dir, "item_corr_stats.csv"))
  }
  
  # --- 4) Behavioral clustering proxy via item-PCA ---
  pca_res <- item_pca(X, max_items = max_items_for_pca)
  if (!is.null(pca_res)) {
    ve <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
    ve_df <- data.frame(PC = seq_along(ve), VarExplained = ve)
    fwrite(ve_df, file.path(out_dir, "item_pca_variance_explained.csv"))
    
    p_scree <- ggplot(ve_df, aes(PC, VarExplained)) +
      geom_line() + geom_point() +
      labs(title = paste0(base, ": item-PCA scree (subsampled items)"),
           x = "PC", y = "Variance explained") +
      theme_minimal()
    ggsave(file.path(out_dir, "item_pca_scree.pdf"), p_scree, width = 7, height = 4)
    
    scores <- pca_res$scores
    if (nrow(scores) >= 200) {
      set.seed(1)
      K <- 5
      km <- kmeans(scores[, c("PC1", "PC2")], centers = K, nstart = 10)
      plot_df <- scores %>% mutate(cluster = factor(km$cluster))
      
      p_scatter <- ggplot(plot_df, aes(PC1, PC2, color = cluster)) +
        geom_point(alpha = 0.5, size = 1.2) +
        labs(title = paste0(base, ": item-PCA (PC1 vs PC2) with kmeans clusters"),
             x = "PC1", y = "PC2") +
        theme_minimal()
      
      ggsave(file.path(out_dir, "item_pca_pc1_pc2_clusters.pdf"), p_scatter, width = 7, height = 5)
    }
  }
  
  # --- File-level summary ---
  summary_row <- data.frame(
    File = f,
    Dataset = dataset,
    Mode = mode,
    n_models = nrow(X),
    n_items = ncol(X),
    
    # Core summaries
    p_item_mean = mean(item_df$p_correct),
    p_item_sd = sd(item_df$p_correct),
    var_item_mean = mean(item_df$var),
    var_item_sd = sd(item_df$var),
    
    # Determinism / extremes (explicit thresholds)
    eps_var = eps_var,
    p_lo = p_lo,
    p_hi = p_hi,
    frac_items_zero_var = mean(item_df$var == 0),
    frac_items_var_lt_eps = mean(item_df$var < eps_var),
    frac_items_p_lt_p_lo = mean(item_df$p_correct < p_lo),
    frac_items_p_gt_p_hi = mean(item_df$p_correct > p_hi)
  )
  
  if (!is.null(corr_stats)) {
    summary_row$corr_median <- corr_stats$corr_median
    summary_row$corr_q95 <- corr_stats$corr_q95
    summary_row$frac_corr_gt_0.95 <- corr_stats$frac_gt_0.95
  } else {
    summary_row$corr_median <- NA_real_
    summary_row$corr_q95 <- NA_real_
    summary_row$frac_corr_gt_0.95 <- NA_real_
  }
  
  summary_row$spearman_idx_p <- rho
  
  all_summaries[[base]] <- summary_row
}


summary_df <- bind_rows(all_summaries)
fwrite(summary_df, file.path(out_root, "matrix_structure_summary_all.csv"))

message("\nSaved master summary to: ", file.path(out_root, "matrix_structure_summary_all.csv"))
message("Done.")

