# === 1_best_fit_FINAL_PHASED.R : Final fitting with fixed n = 500 ===

setwd("E:/Thesis/IRTNet/output/Hypothesis_2/predictions")

library(mirt)
library(dplyr)
library(tidyr)
library(stringr)
library(data.table)
library(tools)
library(purrr)

test_files <- c(
  "ImageNet_zeroshot.csv",
  "ImageNet-C_zeroshot.csv",
  "ImageNet-C_trained.csv",
  "Sketch_zeroshot.csv",
  "Sketch_trained.csv",
  "CIFAR100_zeroshot.csv",
  "CIFAR100_trained.csv"
)

target_n    <- 500
output_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/theta_est"
best_seed_label <- 42

pool_mode <- "informative"

p_extreme_low  <- 0.01
p_extreme_high <- 0.99
var_near_zero  <- 1e-4

# -----------------------------
# Helpers
# -----------------------------
get_pool_params <- function(dataset, mode) {
  p_lower <- 0.20
  p_upper <- 0.80
  var_min <- 0.01
  
  if (dataset == "ImageNet" && mode == "zeroshot") {
    p_lower <- 0.10
    p_upper <- 0.90
    var_min <- 0.005
  }
  if (dataset == "ImageNet-C" && mode == "trained") {
    p_lower <- 0.02
    p_upper <- 0.60
    var_min <- 0.005
  }
  if (dataset == "Sketch" && mode == "zeroshot") {
    p_lower <- 0.05
    p_upper <- 0.85
    var_min <- 0.005
  }
  if (dataset == "CIFAR100" && mode == "trained") {
    p_lower <- 0.10
    p_upper <- 0.90
    var_min <- 0.005
  }
  
  list(p_lower = p_lower, p_upper = p_upper, var_min = var_min)
}

load_binary_data <- function(file) {
  df <- read.csv(file)
  
  binary_data <- df[, -(1:2)]
  binary_data <- as.data.frame(lapply(binary_data, function(x) as.numeric(as.character(x))))
  
  complete_rows <- complete.cases(binary_data)
  binary_data   <- binary_data[complete_rows, ]
  item_names    <- df$Item[complete_rows]
  
  list(binary = binary_data, items = item_names, n_raw_items = nrow(df), n_complete_items = sum(complete_rows))
}

flip_to_models_x_items <- function(binary, items, common_items) {
  idx <- match(common_items, items)
  binary_sub <- binary[idx, , drop = FALSE]
  flipped <- as.data.frame(t(as.matrix(binary_sub)))
  rownames(flipped) <- colnames(binary_sub)
  colnames(flipped) <- common_items
  flipped
}

compute_svd_ratio <- function(X) {
  Xc <- scale(as.matrix(X), center = TRUE, scale = FALSE)
  s <- tryCatch(svd(Xc, nu = 0, nv = 0)$d, error = function(e) NULL)
  if (is.null(s) || length(s) < 2) return(NA_real_)
  as.numeric(s[1] / s[2])
}

item_stats <- function(X) {
  p <- colMeans(X)
  v <- apply(X, 2, var)
  data.frame(
    Item = names(p),
    p = as.numeric(p),
    var = as.numeric(v),
    class = sub("/.*", "", names(p)),
    stringsAsFactors = FALSE
  )
}

define_eligible_pool <- function(stats_df, dataset, modes_present) {
  if (pool_mode == "full") {
    eligible <- stats_df
  } else {
    params <- lapply(modes_present, function(md) get_pool_params(dataset, md))
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
  if (nrow(eligible_df) == 0) stop("Eligible pool is empty. Check thresholds/pool_mode.")
  
  n_classes <- length(unique(eligible_df$class))
  max_per_class <- max(1, floor(1.5 * target_n / n_classes))
  
  eligible_df <- eligible_df %>%
    arrange(abs(p - 0.5), desc(var), class, Item)
  
  chosen <- character(0)
  per_class <- integer(0)
  names(per_class) <- character(0)
  
  for (i in seq_len(nrow(eligible_df))) {
    if (length(chosen) >= target_n) break
    cl  <- eligible_df$class[i]
    itm <- eligible_df$Item[i]
    cur <- if (cl %in% names(per_class)) per_class[cl] else 0L
    if (cur < max_per_class) {
      chosen <- c(chosen, itm)
      per_class[cl] <- cur + 1L
    }
  }
  
  list(chosen = chosen, max_per_class = max_per_class, n_classes = n_classes)
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
      reason = c("var==0", paste0("var<", var_near_zero), paste0("p outside [", p_extreme_low, ",", p_extreme_high, "]")),
      n = c(length(drop_zero), length(drop_near_zero), length(drop_extreme)),
      stringsAsFactors = FALSE
    )
  )
}

eligible_pool_audit <- function(stats_df, dataset, modes_present) {
  n0 <- nrow(stats_df)
  
  # dataset/mode dependent thresholds
  params <- lapply(modes_present, function(md) get_pool_params(dataset, md))
  p_lower <- min(vapply(params, `[[`, numeric(1), "p_lower"))
  p_upper <- max(vapply(params, `[[`, numeric(1), "p_upper"))
  var_min <- min(vapply(params, `[[`, numeric(1), "var_min"))
  
  after_ds <- stats_df %>% filter(p > p_lower, p < p_upper, var >= var_min)
  n1 <- nrow(after_ds)
  
  # global extreme/near-deterministic filter
  after_global <- after_ds %>% filter(p > p_extreme_low, p < p_extreme_high, var >= var_near_zero)
  n2 <- nrow(after_global)
  
  data.frame(
    Dataset = dataset,
    n_start = n0,
    n_after_dataset_thresholds = n1,
    n_after_global_filters = n2,
    dropped_dataset_thresholds = n0 - n1,
    dropped_global_filters = n1 - n2,
    p_lower = p_lower, p_upper = p_upper, var_min = var_min,
    p_extreme_low = p_extreme_low, p_extreme_high = p_extreme_high, var_near_zero = var_near_zero,
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
      method    = "EM",
      technical = list(NCYCLES = 1000),
      TOL       = 1e-3,
      pars      = start_vals
    )
  }, error = function(e) {
    warning("mirt fit failed: ", conditionMessage(e))
    NULL
  })
}

# -----------------------------
# File index
# -----------------------------
file_info <- data.frame(
  file = test_files,
  base = file_path_sans_ext(test_files),
  stringsAsFactors = FALSE
) %>%
  mutate(
    parts   = strsplit(base, "_"),
    dataset = vapply(parts, `[`, character(1), 1),
    mode    = ifelse(str_detect(base, "trained"), "trained", "zeroshot")
  )

datasets <- unique(file_info$dataset)

# Collect dataset-level summary
summary_rows <- list()

cat("\n=============================================\n")
cat("Sampling + fitting run\n")
cat("target_n =", target_n, "| pool_mode =", pool_mode, "\n")
cat("global filters: p in (", p_extreme_low, ",", p_extreme_high, "), var >=", var_near_zero, "\n")
cat("output_root:", output_root, "\n")
cat("=============================================\n")

for (ds in datasets) {
  cat("\n===============================\n")
  cat("Processing dataset:", ds, "\n")
  
  fi_ds <- file_info %>% filter(dataset == ds)
  modes_present <- fi_ds$mode
  
  # Load both modes (if present)
  loaded_list <- lapply(fi_ds$file, load_binary_data)
  
  # Report raw vs complete-case item counts per file (helps interpret n0)
  for (i in seq_along(fi_ds$file)) {
    cat("  File:", fi_ds$file[i],
        "| raw items:", loaded_list[[i]]$n_raw_items,
        "| complete-case items:", loaded_list[[i]]$n_complete_items, "\n")
  }
  
  # Intersection of complete-case items across modes
  common_items <- Reduce(intersect, lapply(loaded_list, `[[`, "items"))
  if (length(common_items) == 0) {
    warning("No common items across modes for dataset ", ds, ". Skipping.")
    next
  }
  cat("Common items across modes (n0):", length(common_items), "\n")
  
  flipped_list <- list()
  for (i in seq_along(fi_ds$file)) {
    md <- fi_ds$mode[i]
    flipped_list[[md]] <- flip_to_models_x_items(
      loaded_list[[i]]$binary,
      loaded_list[[i]]$items,
      common_items
    )
    cat("  Mode", md, "matrix dims (models x items):", nrow(flipped_list[[md]]), "x", ncol(flipped_list[[md]]), "\n")
  }
  
  combined <- do.call(rbind, flipped_list)
  stats_all <- item_stats(combined)
  svd_ratio <- compute_svd_ratio(combined)
  
  audit_elig <- eligible_pool_audit(stats_all, ds, modes_present)
  eligible <- define_eligible_pool(stats_all, ds, modes_present)
  
  cat("Eligible pool audit:\n")
  print(audit_elig[, c("n_start","n_after_dataset_thresholds","n_after_global_filters",
                       "dropped_dataset_thresholds","dropped_global_filters")])
  
  # Save it
  fwrite(audit_elig, file.path(ds_root, "ELIGIBLE_POOL_AUDIT.csv"))
  
  # Save pool audit files
  ds_root <- file.path(output_root, ds)
  dir.create(ds_root, recursive = TRUE, showWarnings = FALSE)
  fwrite(eligible, file.path(ds_root, "Eligible_Pool_AllModes.csv"))
  
  sel <- select_items_with_class_cap(eligible, target_n)
  selected_items <- sel$chosen
  max_per_class <- sel$max_per_class
  n_classes <- sel$n_classes
  
  cat("Selected items after class cap (n3):", length(selected_items),
      "| n_classes:", n_classes,
      "| max_per_class:", max_per_class, "\n")
  
  pool_params_used <- lapply(modes_present, function(md) {
    p <- get_pool_params(ds, md)
    data.frame(Dataset = ds, Mode = md,
               p_lower = p$p_lower, p_upper = p$p_upper, var_min = p$var_min,
               stringsAsFactors = FALSE)
  }) %>% bind_rows()
  fwrite(pool_params_used, file.path(ds_root, "POOL_PARAMS_BY_MODE.csv"))
  
  audit <- data.frame(
    Dataset = ds,
    pool_mode = pool_mode,
    target_n = target_n,
    common_items_n = length(common_items),
    eligible_pool_n = nrow(eligible),
    selected_n = length(selected_items),
    n_classes = n_classes,
    class_cap_max_per_class = max_per_class,
    svd_ratio_1_over_2 = svd_ratio,
    p_extreme_low = p_extreme_low,
    p_extreme_high = p_extreme_high,
    var_near_zero = var_near_zero,
    stringsAsFactors = FALSE
  )
  fwrite(audit, file.path(ds_root, "SELECTION_AUDIT.csv"))
  
  selected_df <- stats_all %>%
    filter(Item %in% selected_items) %>%
    mutate(p_global = p, var_global = var) %>%
    select(Item, class, p_global, var_global)
  fwrite(selected_df, file.path(ds_root, "Selected_Items_AllModes.csv"))
  fwrite(selected_df %>% count(class), file.path(ds_root, "Selected_Items_ClassCounts.csv"))
  
  # Per-mode audit counts
  per_mode_used <- list()
  per_mode_dropped <- list()
  per_mode_drop_breakdown <- list()
  
  for (i in seq_len(nrow(fi_ds))) {
    mode  <- fi_ds$mode[i]
    fname <- fi_ds$file[i]
    
    cat("\n--- Fitting", ds, mode, "from file", fname, "\n")
    
    pr <- prune_items_in_mode(flipped_list[[mode]], selected_items)
    X <- pr$X
    
    per_mode_used[[mode]] <- ncol(X)
    per_mode_dropped[[mode]] <- length(pr$dropped)
    per_mode_drop_breakdown[[mode]] <- pr$drop_breakdown
    
    cat("  After pruning: used (n4) =", ncol(X), "| dropped =", length(pr$dropped), "\n")
    if (nrow(pr$drop_breakdown) > 0) {
      cat("  Drop breakdown:\n")
      print(pr$drop_breakdown)
    }
    
    out_dir <- file.path(output_root, ds, mode, paste0("best_", ncol(X), "_", best_seed_label))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    
    fwrite(pr$drop_breakdown, file.path(out_dir, "Pruning_Summary.csv"))
    if (length(pr$dropped) > 0) {
      fwrite(data.frame(Item = pr$dropped), file.path(out_dir, "Dropped_Items.csv"))
    }
    
    sample_info <- data.frame(
      Item = colnames(X),
      p    = colMeans(X),
      var  = apply(X, 2, var),
      class = sub("/.*", "", colnames(X)),
      stringsAsFactors = FALSE
    )
    fwrite(sample_info, file.path(out_dir, "Sampled_Items.csv"))
    
    if (ncol(X) < 10) {
      warning("Too few informative items (", ncol(X), ") for ", ds, " ", mode, ". Skipping fit.")
      next
    }
    
    fit <- fit_2pl(X)
    if (is.null(fit)) {
      warning("Fit failed for ", ds, " ", mode, ". Skipping outputs.")
      next
    }
    
    converged <- tryCatch(extract.mirt(fit, "converged"), error = function(e) NA)
    loglik_val <- tryCatch(extract.mirt(fit, "logLik"), error = function(e) NA_real_)
    its_val    <- tryCatch(extract.mirt(fit, "iterations"), error = function(e) NA_integer_)
    
    theta_scores <- fscores(fit, full.scores = TRUE)
    rownames(theta_scores) <- rownames(X)
    
    theta_df <- data.frame(
      Model   = rownames(theta_scores),
      Theta   = theta_scores[, 1],
      Dataset = ds
    )
    fwrite(theta_df, file.path(out_dir, "Theta_ModelAbilities_Long.csv"))
    
    item_params_raw <- coef(fit, simplify = TRUE)$items
    saveRDS(item_params_raw, file.path(out_dir, "ItemParameters.rds"))
    
    fit_meta <- data.frame(
      Dataset = ds,
      Mode = mode,
      n_items_used = ncol(X),
      n_items_dropped = length(pr$dropped),
      converged = converged,
      logLik = loglik_val,
      iterations = its_val,
      svd_ratio_1_over_2 = svd_ratio,
      pool_mode = pool_mode,
      stringsAsFactors = FALSE
    )
    fwrite(fit_meta, file.path(out_dir, "Fit_Metadata.csv"))
  }
  
  # Dataset-level summary row for appendix
  summary_rows[[ds]] <- data.frame(
    Dataset = ds,
    common_items_n = length(common_items),             # n0
    eligible_pool_n = nrow(eligible),                  # n2
    selected_n = length(selected_items),               # n3
    n_classes = n_classes,
    class_cap_max_per_class = max_per_class,
    used_items_zeroshot = if ("zeroshot" %in% names(per_mode_used)) per_mode_used[["zeroshot"]] else NA_integer_,  # n4 (ZS)
    used_items_trained  = if ("trained"  %in% names(per_mode_used)) per_mode_used[["trained"]]  else NA_integer_,  # n4 (Tr)
    dropped_items_zeroshot = if ("zeroshot" %in% names(per_mode_dropped)) per_mode_dropped[["zeroshot"]] else NA_integer_,
    dropped_items_trained  = if ("trained"  %in% names(per_mode_dropped)) per_mode_dropped[["trained"]]  else NA_integer_,
    svd_ratio_1_over_2 = svd_ratio,
    stringsAsFactors = FALSE
  )
}

if (length(summary_rows) > 0) {
  summary_tbl <- bind_rows(summary_rows) %>%
    arrange(match(Dataset, c("ImageNet","ImageNet-C","Sketch","CIFAR100")))
  out_csv <- file.path(output_root, "DATASET_LEVEL_FIT_SUMMARY.csv")
  fwrite(summary_tbl, out_csv)
  cat("\nSaved dataset-level sampling summary to:\n", out_csv, "\n")
}

cat("\nAll done.\n")

