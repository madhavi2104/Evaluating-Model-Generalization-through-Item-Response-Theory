# ============================================================
# fit_2pl_best_500.R
# Final 2PL fitting with fixed target_n = 500
# Repo-safe version using merged binary response matrices
# ============================================================

rm(list = ls())

library(mirt)
library(dplyr)
library(stringr)
library(data.table)
library(tools)
library(purrr)

# -----------------------------
# Config
# -----------------------------
repo_root <- "."
input_root <- file.path(repo_root, "results", "response_matrices")
output_root <- file.path(repo_root, "results", "analysis", "irt", "fits_best_500")

dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

target_n <- 500
best_seed_label <- 42L

pool_mode <- "informative"

# global safety filters
p_extreme_low  <- 0.01
p_extreme_high <- 0.99
var_near_zero  <- 1e-4

# dataset order for summaries
dataset_order <- c("ImageNet", "ImageNet-C", "Sketch", "ImageNet-Sketch", "CIFAR100")

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

  tibble(
    matrix_file = matrix_files,
    run_dir = dirname(matrix_file),
    regime = basename(run_dir),
    dataset = basename(dirname(run_dir)),
    item_metadata_file = file.path(run_dir, "item_metadata.csv")
  )
}

read_binary_run <- function(matrix_file, item_metadata_file = NULL) {
  df <- fread(matrix_file)

  if (!"item_id" %in% names(df)) {
    stop("Expected 'item_id' column in: ", matrix_file)
  }

  model_cols <- setdiff(names(df), "item_id")
  if (length(model_cols) == 0) {
    stop("No model columns found in: ", matrix_file)
  }

  binary_data <- df[, ..model_cols]
  binary_data <- as.data.frame(lapply(binary_data, function(x) as.numeric(as.character(x))))

  complete_rows <- complete.cases(binary_data)
  binary_data <- binary_data[complete_rows, , drop = FALSE]
  item_id <- df$item_id[complete_rows]
  item_names <- paste0("item_", item_id)

  flipped <- as.data.frame(t(as.matrix(binary_data)))
  rownames(flipped) <- colnames(binary_data)
  colnames(flipped) <- item_names

  meta_sub <- NULL
  if (!is.null(item_metadata_file) && file.exists(item_metadata_file)) {
    meta <- fread(item_metadata_file)
    if (!"item_id" %in% names(meta)) {
      stop("Expected 'item_id' in item metadata file: ", item_metadata_file)
    }
    meta_sub <- meta[match(item_id, meta$item_id), ]
    meta_sub$Item <- item_names
  }

  list(
    X = flipped,                            # models x items
    items = item_names,
    n_raw_items = nrow(df),
    n_complete_items = sum(complete_rows),
    item_meta = meta_sub
  )
}

get_pool_params <- function(dataset, regime) {
  p_lower <- 0.20
  p_upper <- 0.80
  var_min <- 0.01

  if (dataset == "ImageNet" && regime == "zero_shot") {
    p_lower <- 0.10
    p_upper <- 0.90
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

  list(p_lower = p_lower, p_upper = p_upper, var_min = var_min)
}

compute_svd_ratio <- function(X) {
  Xc <- scale(as.matrix(X), center = TRUE, scale = FALSE)
  s <- tryCatch(svd(Xc, nu = 0, nv = 0)$d, error = function(e) NULL)
  if (is.null(s) || length(s) < 2) return(NA_real_)
  as.numeric(s[1] / s[2])
}

item_stats <- function(X, item_meta = NULL) {
  p <- colMeans(X)
  v <- apply(X, 2, var)

  out <- data.frame(
    Item = names(p),
    p = as.numeric(p),
    var = as.numeric(v),
    stringsAsFactors = FALSE
  )

  if (!is.null(item_meta) && "Item" %in% names(item_meta)) {
    meta_keep <- item_meta[, intersect(
      c("Item", "item_id", "item_path", "true_label_idx", "true_label_name"),
      names(item_meta)
    ), with = FALSE]

    out <- out %>%
      left_join(as.data.frame(meta_keep), by = "Item")
  }

  if (!"true_label_name" %in% names(out)) {
    out$true_label_name <- sub("/.*", "", out$Item)
  }

  out$class <- as.character(out$true_label_name)
  out
}

define_eligible_pool <- function(stats_df, dataset, regimes_present) {
  if (pool_mode == "full") {
    eligible <- stats_df
  } else {
    params <- lapply(regimes_present, function(rg) get_pool_params(dataset, rg))
    p_lower <- min(vapply(params, `[[`, numeric(1), "p_lower"))
    p_upper <- max(vapply(params, `[[`, numeric(1), "p_upper"))
    var_min <- min(vapply(params, `[[`, numeric(1), "var_min"))

    eligible <- stats_df %>%
      filter(p > p_lower, p < p_upper, var >= var_min)
  }

  eligible <- eligible %>%
    filter(p > p_extreme_low, p < p_extreme_high, var >= var_near_zero)

  eligible
}

select_items_with_class_cap <- function(eligible_df, target_n) {
  if (nrow(eligible_df) == 0) {
    stop("Eligible pool is empty. Check thresholds or pool_mode.")
  }

  n_classes <- length(unique(eligible_df$class))
  max_per_class <- max(1, floor(1.5 * target_n / n_classes))

  ranked <- eligible_df %>%
    arrange(abs(p - 0.5), desc(var), class, Item)

  chosen <- character(0)
  per_class <- setNames(integer(n_classes), sort(unique(ranked$class)))

  for (itm in ranked$Item) {
    cl <- ranked$class[match(itm, ranked$Item)]
    cur <- per_class[[cl]]

    if (cur < max_per_class) {
      chosen <- c(chosen, itm)
      per_class[[cl]] <- cur + 1L
    }

    if (length(chosen) >= target_n) break
  }

  if (length(chosen) < min(target_n, nrow(ranked))) {
    remaining <- ranked %>% filter(!Item %in% chosen)
    need <- min(target_n, nrow(ranked)) - length(chosen)
    chosen <- c(chosen, head(remaining$Item, need))
  }

  list(
    chosen = chosen,
    max_per_class = max_per_class,
    n_classes = n_classes
  )
}

prune_items_in_mode <- function(X_mode, selected_items) {
  X <- X_mode[, selected_items, drop = FALSE]

  p_m <- colMeans(X)
  v_m <- apply(X, 2, var)

  drop_zero <- names(v_m)[v_m == 0]
  drop_near_zero <- names(v_m)[v_m < var_near_zero]
  drop_extreme <- names(p_m)[p_m <= p_extreme_low | p_m >= p_extreme_high]

  drop_all <- unique(c(drop_zero, drop_near_zero, drop_extreme))
  keep <- setdiff(colnames(X), drop_all)

  list(
    X = X[, keep, drop = FALSE],
    dropped = drop_all,
    drop_breakdown = data.frame(
      reason = c(
        "var==0",
        paste0("var<", var_near_zero),
        paste0("p outside [", p_extreme_low, ",", p_extreme_high, "]")
      ),
      n = c(length(drop_zero), length(drop_near_zero), length(drop_extreme)),
      stringsAsFactors = FALSE
    )
  )
}

eligible_pool_audit <- function(stats_df, dataset, regimes_present) {
  n0 <- nrow(stats_df)

  params <- lapply(regimes_present, function(rg) get_pool_params(dataset, rg))
  p_lower <- min(vapply(params, `[[`, numeric(1), "p_lower"))
  p_upper <- max(vapply(params, `[[`, numeric(1), "p_upper"))
  var_min <- min(vapply(params, `[[`, numeric(1), "var_min"))

  after_ds <- stats_df %>% filter(p > p_lower, p < p_upper, var >= var_min)
  n1 <- nrow(after_ds)

  after_global <- after_ds %>%
    filter(p > p_extreme_low, p < p_extreme_high, var >= var_near_zero)
  n2 <- nrow(after_global)

  data.frame(
    Dataset = dataset,
    n_start = n0,
    n_after_dataset_thresholds = n1,
    n_after_global_filters = n2,
    dropped_dataset_thresholds = n0 - n1,
    dropped_global_filters = n1 - n2,
    p_lower = p_lower,
    p_upper = p_upper,
    var_min = var_min,
    p_extreme_low = p_extreme_low,
    p_extreme_high = p_extreme_high,
    var_near_zero = var_near_zero,
    stringsAsFactors = FALSE
  )
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

extract_item_params_long <- function(fit) {
  item_params_raw <- coef(fit, simplify = TRUE)$items
  item_params_df <- as.data.frame(item_params_raw)
  item_params_df$Item <- rownames(item_params_df)
  rownames(item_params_df) <- NULL
  item_params_df
}

# -----------------------------
# Main
# -----------------------------
runs <- find_matrix_runs(input_root)
datasets <- unique(runs$dataset)

summary_rows <- list()

cat("\n=============================================\n")
cat("Final 2PL fitting run\n")
cat("target_n =", target_n, "| pool_mode =", pool_mode, "\n")
cat(
  "global filters: p in (", p_extreme_low, ",", p_extreme_high,
  "), var >=", var_near_zero, "\n"
)
cat("input_root:", input_root, "\n")
cat("output_root:", output_root, "\n")
cat("=============================================\n")

for (ds in datasets) {
  cat("\n===============================\n")
  cat("Processing dataset:", ds, "\n")

  fi_ds <- runs %>% filter(dataset == ds)
  regimes_present <- fi_ds$regime

  loaded_list <- lapply(seq_len(nrow(fi_ds)), function(i) {
    read_binary_run(
      matrix_file = fi_ds$matrix_file[i],
      item_metadata_file = fi_ds$item_metadata_file[i]
    )
  })
  names(loaded_list) <- fi_ds$regime

  for (i in seq_along(loaded_list)) {
    cat(
      "  File:", fi_ds$matrix_file[i],
      "| raw items:", loaded_list[[i]]$n_raw_items,
      "| complete-case items:", loaded_list[[i]]$n_complete_items, "\n"
    )
  }

  common_items <- Reduce(intersect, lapply(loaded_list, `[[`, "items"))
  if (length(common_items) == 0) {
    warning("No common items across regimes for dataset ", ds, ". Skipping.")
    next
  }
  cat("Common items across regimes (n0):", length(common_items), "\n")

  flipped_list <- list()
  item_meta_ref <- NULL

  for (rg in names(loaded_list)) {
    ld <- loaded_list[[rg]]
    idx <- match(common_items, ld$items)
    X_sub <- ld$X[, idx, drop = FALSE]
    colnames(X_sub) <- common_items
    flipped_list[[rg]] <- X_sub

    if (!is.null(ld$item_meta)) {
      meta_sub <- as.data.frame(ld$item_meta)
      meta_sub <- meta_sub[match(common_items, meta_sub$Item), ]
      if (is.null(item_meta_ref)) item_meta_ref <- meta_sub
    }
  }

  # Build pooled item stats across all available regimes for this dataset
  pooled_item_stats <- bind_rows(lapply(names(flipped_list), function(rg) {
    X_rg <- flipped_list[[rg]]
    st <- item_stats(X_rg, item_meta = item_meta_ref)
    st$Regime <- rg
    st
  }))

  stats_common <- pooled_item_stats %>%
    group_by(Item) %>%
    summarise(
      p = mean(p, na.rm = TRUE),
      var = mean(var, na.rm = TRUE),
      class = first(class),
      item_id = first(item_id),
      item_path = first(item_path),
      true_label_idx = first(true_label_idx),
      true_label_name = first(true_label_name),
      .groups = "drop"
    )

  audit_tbl <- eligible_pool_audit(stats_common, ds, regimes_present)
  ds_root <- file.path(output_root, ds)
  dir.create(ds_root, recursive = TRUE, showWarnings = FALSE)
  fwrite(audit_tbl, file.path(ds_root, "Eligible_Pool_Audit.csv"))

  eligible <- define_eligible_pool(stats_common, ds, regimes_present)
  cat("Eligible pool after filtering (n2):", nrow(eligible), "\n")

  if (nrow(eligible) == 0) {
    warning("Eligible pool empty for dataset ", ds, ". Skipping.")
    next
  }

  sel <- select_items_with_class_cap(eligible, target_n)
  selected_items <- sel$chosen
  max_per_class <- sel$max_per_class
  n_classes <- sel$n_classes

  cat("Selected items (n3):", length(selected_items), "\n")
  cat("Class cap max_per_class:", max_per_class, "\n")

  selected_df <- eligible %>% filter(Item %in% selected_items)
  fwrite(selected_df, file.path(ds_root, "Selected_Items.csv"))
  fwrite(selected_df %>% count(class), file.path(ds_root, "Selected_Items_ClassCounts.csv"))

  per_regime_used <- list()
  per_regime_dropped <- list()

  for (i in seq_len(nrow(fi_ds))) {
    regime <- fi_ds$regime[i]

    cat("\n--- Fitting", ds, regime, "\n")

    pr <- prune_items_in_mode(flipped_list[[regime]], selected_items)
    X <- pr$X

    per_regime_used[[regime]] <- ncol(X)
    per_regime_dropped[[regime]] <- length(pr$dropped)

    cat("  After pruning: used (n4) =", ncol(X), "| dropped =", length(pr$dropped), "\n")
    if (nrow(pr$drop_breakdown) > 0) {
      cat("  Drop breakdown:\n")
      print(pr$drop_breakdown)
    }

    out_dir <- file.path(output_root, ds, regime, paste0("best_", ncol(X), "_", best_seed_label))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    fwrite(pr$drop_breakdown, file.path(out_dir, "Pruning_Summary.csv"))
    if (length(pr$dropped) > 0) {
      fwrite(data.frame(Item = pr$dropped), file.path(out_dir, "Dropped_Items.csv"))
    }

    sample_info <- selected_df %>%
      filter(Item %in% colnames(X)) %>%
      mutate(
        p = colMeans(X)[match(Item, names(colMeans(X)))],
        var = apply(X, 2, var)[match(Item, names(apply(X, 2, var)))]
      )
    fwrite(sample_info, file.path(out_dir, "Sampled_Items.csv"))

    if (ncol(X) < 10) {
      warning("Too few informative items (", ncol(X), ") for ", ds, " ", regime, ". Skipping fit.")
      next
    }

    svd_ratio <- compute_svd_ratio(X)

    fit <- fit_2pl(X)
    if (is.null(fit)) {
      warning("Fit failed for ", ds, " ", regime, ". Skipping outputs.")
      next
    }

    converged <- tryCatch(extract.mirt(fit, "converged"), error = function(e) NA)
    loglik_val <- tryCatch(as.numeric(extract.mirt(fit, "logLik")), error = function(e) NA_real_)
    its_val <- tryCatch(extract.mirt(fit, "iterations"), error = function(e) NA_integer_)

    theta_scores <- fscores(fit, full.scores = TRUE)
    rownames(theta_scores) <- rownames(X)

    theta_df <- data.frame(
      Model = rownames(theta_scores),
      Theta = theta_scores[, 1],
      Dataset = ds,
      Regime = regime,
      stringsAsFactors = FALSE
    )
    fwrite(theta_df, file.path(out_dir, "Theta_ModelAbilities_Long.csv"))

    item_params_raw <- coef(fit, simplify = TRUE)$items
    saveRDS(item_params_raw, file.path(out_dir, "ItemParameters.rds"))

    item_params_df <- extract_item_params_long(fit)
    item_params_df <- item_params_df %>%
      left_join(selected_df %>% select(any_of(c(
        "Item", "item_id", "item_path", "true_label_idx", "true_label_name", "class"
      ))), by = "Item")
    fwrite(item_params_df, file.path(out_dir, "ItemParameters_Long.csv"))

    fit_meta <- data.frame(
      Dataset = ds,
      Regime = regime,
      n_items_used = ncol(X),
      n_items_dropped = length(pr$dropped),
      converged = converged,
      logLik = loglik_val,
      iterations = its_val,
      svd_ratio_1_over_2 = svd_ratio,
      pool_mode = pool_mode,
      target_n = target_n,
      stringsAsFactors = FALSE
    )
    fwrite(fit_meta, file.path(out_dir, "Fit_Metadata.csv"))
  }

  summary_rows[[ds]] <- data.frame(
    Dataset = ds,
    common_items_n = length(common_items),   # n0
    eligible_pool_n = nrow(eligible),        # n2
    selected_n = length(selected_items),     # n3
    n_classes = n_classes,
    class_cap_max_per_class = max_per_class,
    used_items_zero_shot = if ("zero_shot" %in% names(per_regime_used)) per_regime_used[["zero_shot"]] else NA_integer_,
    used_items_trained = if ("trained" %in% names(per_regime_used)) per_regime_used[["trained"]] else NA_integer_,
    used_items_head_only = if ("head_only" %in% names(per_regime_used)) per_regime_used[["head_only"]] else NA_integer_,
    dropped_items_zero_shot = if ("zero_shot" %in% names(per_regime_dropped)) per_regime_dropped[["zero_shot"]] else NA_integer_,
    dropped_items_trained = if ("trained" %in% names(per_regime_dropped)) per_regime_dropped[["trained"]] else NA_integer_,
    dropped_items_head_only = if ("head_only" %in% names(per_regime_dropped)) per_regime_dropped[["head_only"]] else NA_integer_,
    stringsAsFactors = FALSE
  )
}

if (length(summary_rows) > 0) {
  summary_tbl <- bind_rows(summary_rows) %>%
    mutate(Dataset = factor(Dataset, levels = dataset_order)) %>%
    arrange(Dataset) %>%
    mutate(Dataset = as.character(Dataset))

  out_csv <- file.path(output_root, "DATASET_LEVEL_FIT_SUMMARY.csv")
  fwrite(summary_tbl, out_csv)
  cat("\nSaved dataset-level fitting summary to:\n", out_csv, "\n")
}

cat("\nDone.\n")
