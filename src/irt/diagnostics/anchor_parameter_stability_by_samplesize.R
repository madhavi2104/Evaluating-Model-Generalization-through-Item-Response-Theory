# ============================================================
# anchor_parameter_stability_by_samplesize.R
# Dataset-aware anchor stability across sample sizes
# Repo-safe version for merged binary response matrices
# ============================================================

rm(list = ls())

library(mirt)
library(dplyr)
library(stringr)
library(data.table)
library(tools)
library(ggplot2)
library(readr)

# -----------------------------
# Config
# -----------------------------
repo_root <- "."
input_root <- file.path(repo_root, "results", "response_matrices")
output_root <- file.path(
  repo_root,
  "results",
  "analysis",
  "irt",
  "anchor_stability_informative"
)

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

sample_sizes <- c(20, 50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 2500)
seeds <- c(42, 123, 999)

anchor_n <- 20
pool_mode <- "informative"   # "informative" or "full"

default_p_lower <- 0.2
default_p_upper <- 0.8
default_var_min <- 0.01

eps_b <- 0.05
eps_a <- 0.05

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
      dataset = basename(dirname(run_dir))
    )
}

read_flipped_matrix <- function(matrix_file) {
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
  item_names <- paste0("item_", df$item_id[keep_rows])

  flipped <- as.data.frame(t(as.matrix(X)))
  rownames(flipped) <- colnames(X)
  colnames(flipped) <- item_names

  flipped
}

get_pool_params <- function(dataset, regime) {
  p_lower <- default_p_lower
  p_upper <- default_p_upper
  var_min <- default_var_min

  if (dataset == "ImageNet" && regime == "zero_shot") {
    p_upper <- 0.90
    p_lower <- 0.10
    var_min <- 0.005
  }

  if (dataset == "ImageNet-C" && regime == "trained") {
    p_lower <- 0.02
    p_upper <- 0.60
    var_min <- 0.005
  }

  if ((dataset == "Sketch" || dataset == "ImageNet-Sketch") && regime == "zero_shot") {
    p_lower <- 0.05
    p_upper <- 0.85
    var_min <- 0.005
  }

  if (dataset == "CIFAR100" && regime == "trained") {
    p_lower <- 0.10
    p_upper <- 0.90
    var_min <- 0.005
  }

  if (dataset == "CIFAR100" && regime == "head_only") {
    p_lower <- 0.10
    p_upper <- 0.90
    var_min <- 0.005
  }

  list(
    p_lower = p_lower,
    p_upper = p_upper,
    var_min = var_min
  )
}

make_item_stats <- function(X) {
  p <- colMeans(X)
  v <- apply(X, 2, var)

  data.frame(
    Item = names(p),
    p = as.numeric(p),
    var = as.numeric(v),
    stringsAsFactors = FALSE
  )
}

filter_pool <- function(X, dataset, regime) {
  stats <- make_item_stats(X)

  if (pool_mode == "full") {
    keep <- stats$Item
  } else {
    pars <- get_pool_params(dataset, regime)
    keep <- stats %>%
      filter(p > pars$p_lower, p < pars$p_upper, var >= pars$var_min) %>%
      pull(Item)
  }

  list(
    X = X[, keep, drop = FALSE],
    stats = stats %>% filter(Item %in% keep)
  )
}

select_anchors_stratified <- function(stats_df, anchor_n) {
  df <- stats_df %>% arrange(p)

  n_hard <- floor(anchor_n * 0.25)
  n_easy <- floor(anchor_n * 0.25)
  n_mid  <- anchor_n - n_hard - n_easy

  hard_pool <- df %>% slice_head(n = min(nrow(df), max(50, n_hard * 10)))
  easy_pool <- df %>% slice_tail(n = min(nrow(df), max(50, n_easy * 10)))

  q40 <- as.numeric(quantile(df$p, 0.4, na.rm = TRUE))
  q60 <- as.numeric(quantile(df$p, 0.6, na.rm = TRUE))
  mid_pool <- df %>% filter(p >= q40, p <= q60)

  pick_topvar <- function(pool, n) {
    pool %>%
      arrange(desc(var), Item) %>%
      slice_head(n = n) %>%
      pull(Item)
  }

  anchors <- c(
    pick_topvar(hard_pool, n_hard),
    pick_topvar(mid_pool,  n_mid),
    pick_topvar(easy_pool, n_easy)
  ) %>% unique()

  if (length(anchors) < anchor_n) {
    remaining <- df %>%
      filter(!Item %in% anchors) %>%
      arrange(desc(var), Item)
    anchors <- c(anchors, head(remaining$Item, anchor_n - length(anchors)))
  }

  anchors[1:anchor_n]
}

fit_2pl <- function(X) {
  start_vals <- mirt(X, 1, itemtype = "2PL", pars = "values")
  start_vals$value[start_vals$name == "d"] <- -qlogis(
    pmin(pmax(colMeans(X), 1e-4), 1 - 1e-4)
  )

  tryCatch({
    mirt(
      X, 1, itemtype = "2PL",
      method = "EM",
      technical = list(NCYCLES = 1000),
      TOL = 1e-3,
      pars = start_vals
    )
  }, error = function(e) {
    warning("mirt fit failed: ", conditionMessage(e))
    NULL
  })
}

extract_ab <- function(fit, anchor_items) {
  if (is.null(fit)) return(NULL)

  item_coefs <- coef(fit, IRTpars = TRUE, simplify = TRUE)$items
  cols <- colnames(item_coefs)

  a_col <- if ("a" %in% cols) {
    "a"
  } else if ("a1" %in% cols) {
    "a1"
  } else {
    cols[grepl("^a", cols)][1]
  }

  b_col <- if ("b" %in% cols) {
    "b"
  } else if ("d" %in% cols) {
    "d"
  } else {
    cols[grepl("^b|^d", cols)][1]
  }

  anchors_in_fit <- intersect(anchor_items, rownames(item_coefs))
  if (!length(anchors_in_fit)) return(NULL)

  data.frame(
    Item = anchors_in_fit,
    a = item_coefs[anchors_in_fit, a_col],
    b = item_coefs[anchors_in_fit, b_col],
    stringsAsFactors = FALSE
  )
}

# More robust N* rule:
# first N where median delta < eps and stays below for all later N
compute_threshold_N <- function(df_long, param = "b", eps = 0.05) {
  dd <- df_long %>%
    arrange(Seed, Item, SampleSize) %>%
    group_by(Seed, Item) %>%
    mutate(delta = abs(.data[[param]] - lag(.data[[param]]))) %>%
    ungroup()

  agg <- dd %>%
    filter(!is.na(delta)) %>%
    group_by(SampleSize) %>%
    summarise(med_delta = median(delta, na.rm = TRUE), .groups = "drop") %>%
    arrange(SampleSize)

  if (nrow(agg) == 0) return(list(N_star = NA_integer_, table = agg))

  ok <- agg$med_delta < eps
  if (!any(ok)) return(list(N_star = NA_integer_, table = agg))

  for (k in which(ok)) {
    if (all(ok[k:length(ok)])) {
      return(list(N_star = agg$SampleSize[k], table = agg))
    }
  }

  list(N_star = NA_integer_, table = agg)
}

# -----------------------------
# Main
# -----------------------------
runs <- find_matrix_runs(input_root)

cat("\n=============================================\n")
cat("Anchor stability by sample size\n")
cat("pool_mode =", pool_mode, "\n")
cat("anchor_n  =", anchor_n, "\n")
cat("sample_sizes =", paste(sample_sizes, collapse = ", "), "\n")
cat("seeds =", paste(seeds, collapse = ", "), "\n")
cat("input_root:", input_root, "\n")
cat("output_root:", output_root, "\n")
cat("=============================================\n")

for (i in seq_len(nrow(runs))) {
  matrix_file <- runs$matrix_file[i]
  dataset <- runs$dataset[i]
  regime <- runs$regime[i]

  run_label <- paste(dataset, regime, sep = "_")

  cat("\n=============================\n")
  cat("Processing:", run_label, "\n")
  cat("Matrix file:", matrix_file, "\n")

  out_dir <- file.path(output_root, dataset, regime)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  X_raw <- read_flipped_matrix(matrix_file)
  cat("Raw dims (models x items):", dim(X_raw), "\n")

  pool <- filter_pool(X_raw, dataset, regime)
  X0 <- pool$X
  stats0 <- pool$stats

  cat("Pool dims after mode =", pool_mode, ":", dim(X0), "\n")

  fwrite(stats0, file.path(out_dir, "pool_item_stats.csv"))

  if (ncol(X0) < anchor_n + 10) {
    warning("Too few usable items in pool for ", run_label, ". Skipping.")
    next
  }

  anchors <- select_anchors_stratified(stats0, anchor_n)
  fwrite(data.frame(Item = anchors), file.path(out_dir, "anchor_items.csv"))

  valid_sizes <- sample_sizes[sample_sizes <= ncol(X0)]
  valid_sizes <- valid_sizes[valid_sizes >= anchor_n]

  if (!length(valid_sizes)) {
    warning("No valid sample sizes for ", run_label, ". Skipping.")
    next
  }

  results <- list()

  for (N in valid_sizes) {
    cat("  N =", N, "\n")

    for (sd in seeds) {
      set.seed(sd)

      extras_needed <- N - anchor_n
      remaining <- setdiff(colnames(X0), anchors)
      extras <- if (extras_needed > 0) {
        sample(remaining, extras_needed)
      } else {
        character(0)
      }

      selected <- c(anchors, extras)
      X <- X0[, selected, drop = FALSE]

      v <- apply(X, 2, var)
      X <- X[, v > 0, drop = FALSE]

      anchors_in_sample <- intersect(anchors, colnames(X))
      if (length(anchors_in_sample) < anchor_n) next

      fit <- fit_2pl(X)
      ab <- extract_ab(fit, anchors_in_sample)
      if (is.null(ab)) next

      ab$Dataset <- dataset
      ab$Regime <- regime
      ab$RunLabel <- run_label
      ab$SampleSize <- N
      ab$Seed <- sd

      results[[paste0("N", N, "_S", sd)]] <- ab
    }
  }

  df <- bind_rows(results)

  if (nrow(df) == 0) {
    warning("No valid fits extracted for ", run_label)
    next
  }

  fwrite(df, file.path(out_dir, "anchor_param_trends.csv"))

  p_b <- ggplot(df, aes(x = SampleSize, y = b, group = interaction(Item, Seed))) +
    geom_line(alpha = 0.35) +
    facet_wrap(~ Seed) +
    labs(
      title = paste0(run_label, " | Anchor difficulty stability (b)"),
      subtitle = paste0("pool_mode=", pool_mode, " | ", anchor_n, " anchors"),
      x = "Sample size (items)",
      y = "b (difficulty)"
    ) +
    theme_minimal()

  ggsave(
    file.path(out_dir, "anchor_b_trends_by_seed.pdf"),
    p_b,
    width = 10,
    height = 5
  )

  p_a <- ggplot(df, aes(x = SampleSize, y = a, group = interaction(Item, Seed))) +
    geom_line(alpha = 0.35) +
    facet_wrap(~ Seed) +
    labs(
      title = paste0(run_label, " | Anchor discrimination stability (a)"),
      subtitle = paste0("pool_mode=", pool_mode, " | ", anchor_n, " anchors"),
      x = "Sample size (items)",
      y = "a (discrimination)"
    ) +
    theme_minimal()

  ggsave(
    file.path(out_dir, "anchor_a_trends_by_seed.pdf"),
    p_a,
    width = 10,
    height = 5
  )

  thr_b <- compute_threshold_N(df, param = "b", eps = eps_b)
  thr_a <- compute_threshold_N(df, param = "a", eps = eps_a)

  fwrite(thr_b$table, file.path(out_dir, "threshold_table_b.csv"))
  fwrite(thr_a$table, file.path(out_dir, "threshold_table_a.csv"))

  fwrite(
    data.frame(
      Dataset = dataset,
      Regime = regime,
      RunLabel = run_label,
      pool_mode = pool_mode,
      eps_b = eps_b,
      N_star_b = thr_b$N_star,
      eps_a = eps_a,
      N_star_a = thr_a$N_star,
      stringsAsFactors = FALSE
    ),
    file.path(out_dir, "N_star_thresholds.csv")
  )

  cat("Saved outputs to:", out_dir, "\n")
}

cat("\nAll done.\n")
