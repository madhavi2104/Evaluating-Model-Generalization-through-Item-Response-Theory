# ============================================================
# matrix_structure_diagnostics.R
# Investigate response matrix structure BEFORE IRT fitting
# Repo-safe version for merged binary response matrices
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
repo_root <- "."
input_root <- file.path(repo_root, "results", "response_matrices")
out_root <- file.path(repo_root, "results", "analysis", "irt", "matrix_structure_diagnostics")

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# Correlation / PCA limits for speed
max_items_for_pca <- 5000
corr_item_sample  <- 800
corr_model_min    <- 20

# Near-deterministic thresholds
eps_var <- 1e-6
p_lo <- 0.05
p_hi <- 0.95

# Ordering diagnostics params
rolling_window <- 200
bin_size <- 200

# -----------------------------
# Helpers
# -----------------------------

find_matrix_runs <- function(input_root) {
  matrix_files <- list.files(
    input_root,
    pattern = "^binary_response_matrix\\.csv$",
    recursive = TRUE,
    full.names = TRUE
  )

  if (length(matrix_files) == 0) {
    stop("No binary_response_matrix.csv files found under: ", input_root)
  }

  data.frame(
    matrix_file = matrix_files,
    stringsAsFactors = FALSE
  ) %>%
    mutate(
      run_dir = dirname(matrix_file),
      regime = basename(run_dir),
      dataset = basename(dirname(run_dir)),
      item_metadata_file = file.path(run_dir, "item_metadata.csv")
    )
}

read_response_matrix <- function(matrix_file, item_metadata_file = NULL) {
  df <- fread(matrix_file)

  if (!"item_id" %in% names(df)) {
    stop("Expected 'item_id' column in: ", matrix_file)
  }

  model_cols <- setdiff(names(df), "item_id")
  if (length(model_cols) == 0) {
    stop("No model columns found in: ", matrix_file)
  }

  X <- df[, ..model_cols]
  X <- as.data.frame(lapply(X, function(x) as.numeric(as.character(x))))

  keep_rows <- complete.cases(X)
  X <- X[keep_rows, , drop = FALSE]
  item_id <- df$item_id[keep_rows]

  group <- rep(NA_character_, length(item_id))
  raw_meta <- NULL

  if (!is.null(item_metadata_file) && file.exists(item_metadata_file)) {
    meta <- fread(item_metadata_file)
    raw_meta <- meta

    if (!"item_id" %in% names(meta)) {
      stop("Expected 'item_id' column in item metadata file: ", item_metadata_file)
    }

    meta <- meta[match(item_id, meta$item_id), ]

    if ("true_label_name" %in% names(meta)) {
      group <- as.character(meta$true_label_name)
    } else if ("true_label_idx" %in% names(meta)) {
      group <- as.character(meta$true_label_idx)
    }
  }

  # Convert to models x items
  X_mat <- t(as.matrix(X))

  colnames(X_mat) <- paste0("item_", item_id)
  rownames(X_mat) <- model_cols

  list(
    X = X_mat,                 # models x items
    item_id = item_id,
    group = group,
    raw_meta = raw_meta
  )
}

compute_running_group_coverage <- function(group_vec, window = 200) {
  if (all(is.na(group_vec))) return(NULL)

  n <- length(group_vec)
  if (n < window) window <- max(10, floor(n / 5))

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
  M <- nrow(X)
  I <- ncol(X)

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
runs <- find_matrix_runs(input_root)
all_summaries <- list()

for (i in seq_len(nrow(runs))) {
  matrix_file <- runs$matrix_file[i]
  dataset <- runs$dataset[i]
  regime <- runs$regime[i]
  item_metadata_file <- runs$item_metadata_file[i]

  run_label <- paste(dataset, regime, sep = "_")

  message("\n============================")
  message("Processing: ", run_label)
  message("Matrix file: ", matrix_file)

  out_dir <- file.path(out_root, dataset, regime)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  dat <- read_response_matrix(
    matrix_file = matrix_file,
    item_metadata_file = item_metadata_file
  )

  X <- dat$X
  group <- dat$group

  message("Matrix dims (models x items): ", nrow(X), " x ", ncol(X))

  # --- 1) Implicit ordering check: group coverage ---
  run_cov <- compute_running_group_coverage(group, window = rolling_window)

  if (!is.null(run_cov)) {
    p_cov <- ggplot(run_cov, aes(idx, cum_unique_groups)) +
      geom_line() +
      labs(
        title = paste0(run_label, ": cumulative unique groups over item order"),
        x = "Item index (original order)",
        y = "Cumulative unique groups"
      ) +
      theme_minimal()

    ggsave(file.path(out_dir, "ordering_cum_group_coverage.pdf"), p_cov, width = 8, height = 4)

    p_win <- ggplot(run_cov, aes(idx, win_unique_groups)) +
      geom_line() +
      labs(
        title = paste0(run_label, ": unique groups in rolling window"),
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
  item_df$is_near_det <- item_df$var < eps_var

  fwrite(item_df, file.path(out_dir, "item_variation.csv"))

  p_hist_p <- ggplot(item_df, aes(p_correct)) +
    geom_histogram(bins = 50) +
    geom_vline(xintercept = c(p_lo, 0.5, p_hi), linetype = "dashed") +
    labs(
      title = paste0(run_label, ": item mean correctness"),
      x = "p(correct) across models",
      y = "count"
    ) +
    theme_minimal()
  ggsave(file.path(out_dir, "hist_item_pcorrect.pdf"), p_hist_p, width = 7, height = 4)

  p_hist_v <- ggplot(item_df, aes(var)) +
    geom_histogram(bins = 50) +
    labs(
      title = paste0(run_label, ": item variance"),
      x = "var across models",
      y = "count"
    ) +
    theme_minimal()
  ggsave(file.path(out_dir, "hist_item_var.pdf"), p_hist_v, width = 7, height = 4)

  # --- 2b) Difficulty ordering: p(correct) vs item index ---
  k <- min(rolling_window, nrow(item_df))
  item_df$p_roll <- rolling_mean(item_df$p_correct, k)

  rho <- suppressWarnings(cor(item_df$idx, item_df$p_correct, method = "spearman"))
  fwrite(
    data.frame(
      Dataset = dataset,
      Regime = regime,
      spearman_idx_p = rho,
      rolling_window = k
    ),
    file.path(out_dir, "ordering_index_pcorrect_corr.csv")
  )

  p_order <- ggplot(item_df, aes(idx, p_correct)) +
    geom_point(alpha = 0.25, size = 0.7) +
    geom_line(aes(y = p_roll), linewidth = 0.7) +
    labs(
      title = paste0(run_label, ": p(correct) vs item index"),
      subtitle = paste0(
        "Spearman(idx, p) = ", round(rho, 3),
        " | rolling window = ", k
      ),
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
      title = paste0(run_label, ": binned p(correct) vs item index"),
      subtitle = paste0(
        "bin_size = ", min(bin_size, nrow(item_df)),
        " | ribbon = ±1 SD within bin"
      ),
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
      geom_line() +
      geom_point() +
      labs(
        title = paste0(run_label, ": item-PCA scree (subsampled items)"),
        x = "PC",
        y = "Variance explained"
      ) +
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
        labs(
          title = paste0(run_label, ": item-PCA (PC1 vs PC2) with kmeans clusters"),
          x = "PC1",
          y = "PC2"
        ) +
        theme_minimal()

      ggsave(file.path(out_dir, "item_pca_pc1_pc2_clusters.pdf"), p_scatter, width = 7, height = 5)
    }
  }

  # --- File-level summary ---
  summary_row <- data.frame(
    Dataset = dataset,
    Regime = regime,
    n_models = nrow(X),
    n_items = ncol(X),

    p_item_mean = mean(item_df$p_correct),
    p_item_sd = sd(item_df$p_correct),
    var_item_mean = mean(item_df$var),
    var_item_sd = sd(item_df$var),

    eps_var = eps_var,
    p_lo = p_lo,
    p_hi = p_hi,
    frac_items_zero_var = mean(item_df$var == 0),
    frac_items_var_lt_eps = mean(item_df$var < eps_var),
    frac_items_p_lt_p_lo = mean(item_df$p_correct < p_lo),
    frac_items_p_gt_p_hi = mean(item_df$p_correct > p_hi),

    spearman_idx_p = rho
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

  all_summaries[[run_label]] <- summary_row
}

summary_df <- bind_rows(all_summaries)
fwrite(summary_df, file.path(out_root, "matrix_structure_summary_all.csv"))

message("\nSaved master summary to: ", file.path(out_root, "matrix_structure_summary_all.csv"))
message("Done.")
