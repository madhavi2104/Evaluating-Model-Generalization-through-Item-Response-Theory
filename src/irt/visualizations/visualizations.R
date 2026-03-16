# ============================================================
# visualizations.R
# Repo-safe downstream visualizations for IRTNet thesis
# ============================================================

rm(list = ls())

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(data.table)
library(stringr)
library(purrr)
library(cowplot)
library(patchwork)

# ------------------------------------------------
# Config
# ------------------------------------------------
repo_root <- "."
summary_root <- file.path(repo_root, "results", "analysis", "irt", "summary_tables")
fits_root    <- file.path(repo_root, "results", "analysis", "irt", "fits_best_500")
pred_root    <- file.path(repo_root, "results", "predictions")
out_root     <- file.path(repo_root, "results", "analysis", "irt", "visualizations")

dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

dataset_order <- c("ImageNet", "ImageNet-C", "Sketch", "ImageNet-Sketch", "CIFAR100")

# ------------------------------------------------
# Helpers
# ------------------------------------------------

safe_read_csv <- function(path) {
  if (!file.exists(path)) return(NULL)
  read_csv(path, show_col_types = FALSE)
}

safe_read_rds <- function(path) {
  if (!file.exists(path)) return(NULL)
  readRDS(path)
}

make_numeric_df <- function(df) {
  if (is.null(df)) return(NULL)
  for (nm in names(df)[-1]) {
    df[[nm]] <- suppressWarnings(as.numeric(df[[nm]]))
  }
  df
}

clean_mode_name <- function(x) {
  x <- gsub("^Theta__", "", x)
  x <- gsub("^Accuracy__", "", x)
  x <- gsub("_zero_shot$", "", x)
  x <- gsub("_trained$", "", x)
  x <- gsub("_head_only$", "", x)
  x
}

mode_label <- function(dataset, regime) {
  paste(dataset, regime, sep = "_")
}

find_best_fit_dir <- function(dataset, regime, fits_root) {
  base_dir <- file.path(fits_root, dataset, regime)
  if (!dir.exists(base_dir)) return(NULL)

  cand <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  cand <- cand[grepl("^best_", basename(cand))]
  if (length(cand) == 0) return(NULL)

  # Prefer seed 42 if present, otherwise first
  cand42 <- cand[grepl("_42$", basename(cand))]
  if (length(cand42) > 0) return(cand42[1])

  cand[1]
}

extract_item_params <- function(dataset, regime, fits_root) {
  best_dir <- find_best_fit_dir(dataset, regime, fits_root)
  if (is.null(best_dir)) return(NULL)

  long_file <- file.path(best_dir, "ItemParameters_Long.csv")
  if (file.exists(long_file)) {
    ip <- read_csv(long_file, show_col_types = FALSE)
  } else {
    rds_file <- file.path(best_dir, "ItemParameters.rds")
    ip_raw <- safe_read_rds(rds_file)
    if (is.null(ip_raw)) return(NULL)
    ip <- as.data.frame(ip_raw)
    ip$Item <- rownames(ip)
  }

  if ("d" %in% names(ip) && !"b" %in% names(ip)) {
    names(ip)[names(ip) == "d"] <- "b"
  }

  a_col <- if ("a" %in% names(ip)) {
    "a"
  } else if ("a1" %in% names(ip)) {
    "a1"
  } else {
    NA_character_
  }

  if (!"b" %in% names(ip)) return(NULL)

  ip <- ip %>%
    mutate(
      Dataset = dataset,
      Regime = regime,
      b_rank = rank(b, ties.method = "average") / dplyr::n(),
      b_z = as.numeric(scale(b))
    )

  if (!is.na(a_col) && a_col != "a") {
    ip$a <- ip[[a_col]]
  }

  ip
}

load_prediction_long <- function(dataset, regime, model, pred_root) {
  pred_file <- file.path(pred_root, dataset, regime, model, "binary_correctness.csv")
  if (!file.exists(pred_file)) return(NULL)

  df <- read_csv(pred_file, show_col_types = FALSE)
  req <- c("item_id", "correct")
  if (!all(req %in% names(df))) return(NULL)

  df %>%
    mutate(
      Model = model,
      Dataset = dataset,
      Regime = regime,
      Item = paste0("item_", item_id)
    )
}

compute_rank_variance <- function(df_wide, mode_label_text) {
  df_wide %>%
    pivot_longer(-Model, names_to = "DatasetMode", values_to = "Rank") %>%
    group_by(Model) %>%
    summarise(
      Var = var(Rank, na.rm = TRUE),
      Range = max(Rank, na.rm = TRUE) - min(Rank, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(Var)) %>%
    mutate(Mode = mode_label_text)
}

get_top_models <- function(var_df, top_n = 10) {
  list(
    stable = var_df %>% arrange(Var, Model) %>% slice_head(n = top_n),
    volatile = var_df %>% arrange(desc(Var), Model) %>% slice_head(n = top_n)
  )
}

make_top_slope <- function(df_wide, top_models, title, out_file) {
  if (is.null(df_wide) || nrow(df_wide) == 0) return(NULL)

  df_sub <- df_wide %>% filter(Model %in% top_models$Model)
  if (nrow(df_sub) == 0) return(NULL)

  df_long <- df_sub %>%
    pivot_longer(-Model, names_to = "DatasetMode", values_to = "Rank")

  p <- ggplot(df_long, aes(x = DatasetMode, y = Rank, group = Model, color = Model)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 3, shape = 21, fill = "white", stroke = 1.1) +
    scale_y_reverse() +
    theme_minimal(base_size = 12) +
    labs(title = title, y = "Rank (lower = better)", x = "") +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "right"
    )

  ggsave(out_file, plot = p, width = 9, height = 6, dpi = 300)
  p
}

make_scatter <- function(x, y, xlab, ylab, rho_val, title_text = "") {
  df <- data.frame(x = x, y = y)
  ggplot(df, aes(x, y)) +
    geom_point(size = 3, alpha = 0.8) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
    theme_minimal(base_size = 12) +
    labs(
      title = title_text,
      subtitle = paste0("Spearman ρ = ", round(rho_val, 3)),
      x = xlab,
      y = ylab
    )
}

# ------------------------------------------------
# Load summary tables
# ------------------------------------------------
theta_wide_zero   <- safe_read_csv(file.path(summary_root, "Theta_Wide_ZeroShot.csv"))
theta_rank_zero   <- safe_read_csv(file.path(summary_root, "Theta_Rank_Wide_ZeroShot.csv"))
acc_wide_zero     <- safe_read_csv(file.path(summary_root, "Accuracy_Wide_ZeroShot.csv"))
acc_rank_zero     <- safe_read_csv(file.path(summary_root, "Accuracy_Rank_Wide_ZeroShot.csv"))

theta_wide_train  <- safe_read_csv(file.path(summary_root, "Theta_Wide_TrainedLike.csv"))
theta_rank_train  <- safe_read_csv(file.path(summary_root, "Theta_Rank_Wide_TrainedLike.csv"))
acc_wide_train    <- safe_read_csv(file.path(summary_root, "Accuracy_Wide_TrainedLike.csv"))
acc_rank_train    <- safe_read_csv(file.path(summary_root, "Accuracy_Rank_Wide_TrainedLike.csv"))

theta_wide_zero   <- make_numeric_df(theta_wide_zero)
theta_rank_zero   <- make_numeric_df(theta_rank_zero)
acc_wide_zero     <- make_numeric_df(acc_wide_zero)
acc_rank_zero     <- make_numeric_df(acc_rank_zero)

theta_wide_train  <- make_numeric_df(theta_wide_train)
theta_rank_train  <- make_numeric_df(theta_rank_train)
acc_wide_train    <- make_numeric_df(acc_wide_train)
acc_rank_train    <- make_numeric_df(acc_rank_train)

# ------------------------------------------------
# 1) H1 scatter-grid plots
# ------------------------------------------------
make_hypothesis_plot <- function(hypothesis, out_file) {
  if (is.null(acc_wide_zero) || is.null(theta_wide_zero)) return(NULL)

  # zero-shot datasets excluding ImageNet for target side
  zero_cols_acc   <- setdiff(names(acc_wide_zero), "Model")
  zero_cols_theta <- setdiff(names(theta_wide_zero), "Model")

  zero_targets <- setdiff(intersect(
    gsub("^Theta__", "", zero_cols_theta),
    gsub("^Accuracy__", "", zero_cols_acc)
  ), "ImageNet_zero_shot")

  plots_zero <- list()
  plots_mix  <- list()

  for (ds_mode in zero_targets) {
    ds_name <- gsub("_zero_shot$", "", ds_mode)

    # zero-shot references
    if (hypothesis == "H1.1" && ds_mode %in% zero_cols_acc && "ImageNet_zero_shot" %in% zero_cols_acc) {
      rho0 <- cor(acc_wide_zero$ImageNet_zero_shot, acc_wide_zero[[ds_mode]], method = "spearman", use = "pairwise.complete.obs")
      plots_zero[[ds_mode]] <- make_scatter(
        acc_wide_zero$ImageNet_zero_shot, acc_wide_zero[[ds_mode]],
        "ImageNet acc", paste0(ds_name, " acc"), rho0
      )
    }

    if (hypothesis == "H1.2" && ds_mode %in% zero_cols_acc && ds_mode %in% zero_cols_theta) {
      rho0 <- cor(theta_wide_zero[[ds_mode]], acc_wide_zero[[ds_mode]], method = "spearman", use = "pairwise.complete.obs")
      plots_zero[[ds_mode]] <- make_scatter(
        theta_wide_zero[[ds_mode]], acc_wide_zero[[ds_mode]],
        paste0(ds_name, " θ"), paste0(ds_name, " acc"), rho0
      )
    }

    if (hypothesis == "H1.3" && ds_mode %in% zero_cols_theta && "ImageNet_zero_shot" %in% zero_cols_theta) {
      rho0 <- cor(theta_wide_zero$ImageNet_zero_shot, theta_wide_zero[[ds_mode]], method = "spearman", use = "pairwise.complete.obs")
      plots_zero[[ds_mode]] <- make_scatter(
        theta_wide_zero$ImageNet_zero_shot, theta_wide_zero[[ds_mode]],
        "ImageNet θ", paste0(ds_name, " θ"), rho0
      )
    }

    if (hypothesis == "H1.4" && ds_mode %in% zero_cols_theta && "ImageNet_zero_shot" %in% zero_cols_acc) {
      rho0 <- cor(acc_wide_zero$ImageNet_zero_shot, theta_wide_zero[[ds_mode]], method = "spearman", use = "pairwise.complete.obs")
      plots_zero[[ds_mode]] <- make_scatter(
        acc_wide_zero$ImageNet_zero_shot, theta_wide_zero[[ds_mode]],
        "ImageNet acc", paste0(ds_name, " θ"), rho0
      )
    }

    if (hypothesis == "H1.5" && ds_mode %in% zero_cols_acc && "ImageNet_zero_shot" %in% zero_cols_theta) {
      rho0 <- cor(theta_wide_zero$ImageNet_zero_shot, acc_wide_zero[[ds_mode]], method = "spearman", use = "pairwise.complete.obs")
      plots_zero[[ds_mode]] <- make_scatter(
        theta_wide_zero$ImageNet_zero_shot, acc_wide_zero[[ds_mode]],
        "ImageNet θ", paste0(ds_name, " acc"), rho0
      )
    }

    # trained-like plots only if matching trained/head_only columns exist
    if (!is.null(acc_wide_train) && !is.null(theta_wide_train)) {
      train_candidates_acc   <- names(acc_wide_train)
      train_candidates_theta <- names(theta_wide_train)

      # map dataset name to trained/head_only mode if present
      ds_train_acc <- train_candidates_acc[grepl(paste0("^", ds_name, "_"), train_candidates_acc)]
      ds_train_theta <- train_candidates_theta[grepl(paste0("^", ds_name, "_"), train_candidates_theta)]
      img_train_acc <- train_candidates_acc[grepl("^ImageNet_", train_candidates_acc)]
      img_train_theta <- train_candidates_theta[grepl("^ImageNet_", train_candidates_theta)]

      ds_acc_col <- if (length(ds_train_acc) > 0) ds_train_acc[1] else NA_character_
      ds_theta_col <- if (length(ds_train_theta) > 0) ds_train_theta[1] else NA_character_
      img_acc_col <- if (length(img_train_acc) > 0) img_train_acc[1] else NA_character_
      img_theta_col <- if (length(img_train_theta) > 0) img_train_theta[1] else NA_character_

      if (hypothesis == "H1.1" && !is.na(img_acc_col) && !is.na(ds_acc_col)) {
        rhoM <- cor(acc_wide_train[[img_acc_col]], acc_wide_train[[ds_acc_col]], method = "spearman", use = "pairwise.complete.obs")
        plots_mix[[ds_name]] <- make_scatter(
          acc_wide_train[[img_acc_col]], acc_wide_train[[ds_acc_col]],
          "ImageNet acc", paste0(ds_name, " acc"), rhoM
        )
      }

      if (hypothesis == "H1.2" && !is.na(ds_theta_col) && !is.na(ds_acc_col)) {
        rhoM <- cor(theta_wide_train[[ds_theta_col]], acc_wide_train[[ds_acc_col]], method = "spearman", use = "pairwise.complete.obs")
        plots_mix[[ds_name]] <- make_scatter(
          theta_wide_train[[ds_theta_col]], acc_wide_train[[ds_acc_col]],
          paste0(ds_name, " θ"), paste0(ds_name, " acc"), rhoM
        )
      }

      if (hypothesis == "H1.3" && !is.na(img_theta_col) && !is.na(ds_theta_col)) {
        rhoM <- cor(theta_wide_train[[img_theta_col]], theta_wide_train[[ds_theta_col]], method = "spearman", use = "pairwise.complete.obs")
        plots_mix[[ds_name]] <- make_scatter(
          theta_wide_train[[img_theta_col]], theta_wide_train[[ds_theta_col]],
          "ImageNet θ", paste0(ds_name, " θ"), rhoM
        )
      }

      if (hypothesis == "H1.4" && !is.na(img_acc_col) && !is.na(ds_theta_col)) {
        rhoM <- cor(acc_wide_train[[img_acc_col]], theta_wide_train[[ds_theta_col]], method = "spearman", use = "pairwise.complete.obs")
        plots_mix[[ds_name]] <- make_scatter(
          acc_wide_train[[img_acc_col]], theta_wide_train[[ds_theta_col]],
          "ImageNet acc", paste0(ds_name, " θ"), rhoM
        )
      }

      if (hypothesis == "H1.5" && !is.na(img_theta_col) && !is.na(ds_acc_col)) {
        rhoM <- cor(theta_wide_train[[img_theta_col]], acc_wide_train[[ds_acc_col]], method = "spearman", use = "pairwise.complete.obs")
        plots_mix[[ds_name]] <- make_scatter(
          theta_wide_train[[img_theta_col]], acc_wide_train[[ds_acc_col]],
          "ImageNet θ", paste0(ds_name, " acc"), rhoM
        )
      }
    }
  }

  if (length(plots_zero) == 0 && length(plots_mix) == 0) return(NULL)

  row_zero <- if (length(plots_zero) > 0) wrap_plots(plots_zero, ncol = 3) else NULL
  row_mix  <- if (length(plots_mix) > 0) wrap_plots(plots_mix, ncol = 3) else NULL

  parts <- list()
  if (!is.null(row_zero)) {
    parts[[length(parts) + 1]] <- plot_grid(
      ggdraw() + draw_label("Zero-shot", fontface = "bold", size = 16),
      row_zero,
      ncol = 1,
      rel_heights = c(0.08, 1)
    )
  }
  if (!is.null(row_mix)) {
    parts[[length(parts) + 1]] <- plot_grid(
      ggdraw() + draw_label("Trained / Head-only", fontface = "bold", size = 16),
      row_mix,
      ncol = 1,
      rel_heights = c(0.08, 1)
    )
  }

  grid <- wrap_plots(parts, ncol = 1)
  ggsave(out_file, grid, width = 15, height = 10, dpi = 300)
  grid
}

make_hypothesis_plot("H1.1", file.path(out_root, "H1.1_grid.png"))
make_hypothesis_plot("H1.2", file.path(out_root, "H1.2_grid.png"))
make_hypothesis_plot("H1.3", file.path(out_root, "H1.3_grid.png"))
make_hypothesis_plot("H1.4", file.path(out_root, "H1.4_grid.png"))
make_hypothesis_plot("H1.5", file.path(out_root, "H1.5_grid.png"))

# ------------------------------------------------
# 2) Stable / volatile rank plots
# ------------------------------------------------
var_zero_acc   <- if (!is.null(acc_rank_zero)) compute_rank_variance(acc_rank_zero, "Zero-shot Accuracy") else NULL
var_zero_theta <- if (!is.null(theta_rank_zero)) compute_rank_variance(theta_rank_zero, "Zero-shot Theta") else NULL
var_train_acc  <- if (!is.null(acc_rank_train)) compute_rank_variance(acc_rank_train, "TrainedLike Accuracy") else NULL
var_train_theta <- if (!is.null(theta_rank_train)) compute_rank_variance(theta_rank_train, "TrainedLike Theta") else NULL

var_all <- bind_rows(var_zero_acc, var_zero_theta, var_train_acc, var_train_theta)
if (nrow(var_all) > 0) {
  write_csv(var_all, file.path(out_root, "rank_volatility_summary.csv"))
}

top_zero_acc    <- if (!is.null(var_zero_acc)) get_top_models(var_zero_acc) else NULL
top_zero_theta  <- if (!is.null(var_zero_theta)) get_top_models(var_zero_theta) else NULL
top_train_acc   <- if (!is.null(var_train_acc)) get_top_models(var_train_acc) else NULL
top_train_theta <- if (!is.null(var_train_theta)) get_top_models(var_train_theta) else NULL

if (!is.null(acc_rank_zero) && !is.null(top_zero_acc)) {
  make_top_slope(acc_rank_zero, top_zero_acc$stable,
                 "Top-10 Stable Models (Zero-shot Accuracy)",
                 file.path(out_root, "slope_stable_zero_acc.png"))
  make_top_slope(acc_rank_zero, top_zero_acc$volatile,
                 "Top-10 Volatile Models (Zero-shot Accuracy)",
                 file.path(out_root, "slope_volatile_zero_acc.png"))
}

if (!is.null(theta_rank_zero) && !is.null(top_zero_theta)) {
  make_top_slope(theta_rank_zero, top_zero_theta$stable,
                 "Top-10 Stable Models (Zero-shot Theta)",
                 file.path(out_root, "slope_stable_zero_theta.png"))
  make_top_slope(theta_rank_zero, top_zero_theta$volatile,
                 "Top-10 Volatile Models (Zero-shot Theta)",
                 file.path(out_root, "slope_volatile_zero_theta.png"))
}

if (!is.null(acc_rank_train) && !is.null(top_train_acc)) {
  make_top_slope(acc_rank_train, top_train_acc$stable,
                 "Top-10 Stable Models (Trained-like Accuracy)",
                 file.path(out_root, "slope_stable_train_acc.png"))
  make_top_slope(acc_rank_train, top_train_acc$volatile,
                 "Top-10 Volatile Models (Trained-like Accuracy)",
                 file.path(out_root, "slope_volatile_train_acc.png"))
}

if (!is.null(theta_rank_train) && !is.null(top_train_theta)) {
  make_top_slope(theta_rank_train, top_train_theta$stable,
                 "Top-10 Stable Models (Trained-like Theta)",
                 file.path(out_root, "slope_stable_train_theta.png"))
  make_top_slope(theta_rank_train, top_train_theta$volatile,
                 "Top-10 Volatile Models (Trained-like Theta)",
                 file.path(out_root, "slope_volatile_train_theta.png"))
}

# ------------------------------------------------
# 3) CCC plots
# ------------------------------------------------
compute_ccc <- function(dataset, regime, models_to_plot, global_breaks = NULL,
                        stable_models = NULL, volatile_models = NULL) {
  ip <- extract_item_params(dataset, regime, fits_root)
  if (is.null(ip)) return(NULL)
  if (!all(c("Item", "b", "b_rank") %in% names(ip))) return(NULL)

  ccc_all <- list()

  for (m in models_to_plot) {
    pred_df <- load_prediction_long(dataset, regime, m, pred_root)
    if (is.null(pred_df)) next

    merged <- pred_df %>%
      left_join(ip %>% select(Item, b, b_rank, b_z), by = "Item") %>%
      filter(!is.na(b_rank))

    if (nrow(merged) == 0) next

    if (is.null(global_breaks)) {
      qbreaks <- unique(quantile(merged$b_rank, probs = seq(0, 1, 0.2), na.rm = TRUE))
    } else {
      qbreaks <- unique(global_breaks)
    }

    if (length(qbreaks) < 2) next

    merged <- merged %>%
      mutate(
        DifficultyBin = cut(
          pmax(pmin(b_rank, max(qbreaks)), min(qbreaks)),
          breaks = qbreaks,
          include.lowest = TRUE,
          right = FALSE,
          dig.lab = 3
        )
      )

    acc_by_bin <- merged %>%
      group_by(DifficultyBin) %>%
      summarise(
        MeanDifficulty = mean(b, na.rm = TRUE),
        Accuracy = mean(correct, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(
        Model = tolower(m),
        Group = case_when(
          tolower(m) %in% stable_models ~ "Stable",
          tolower(m) %in% volatile_models ~ "Volatile",
          TRUE ~ "Other"
        )
      )

    ccc_all[[m]] <- acc_by_bin
  }

  bind_rows(ccc_all)
}

plot_ccc <- function(dataset, regime, models_to_plot, global_breaks = NULL,
                     stable_models = NULL, volatile_models = NULL) {
  ccc_df <- compute_ccc(
    dataset = dataset,
    regime = regime,
    models_to_plot = models_to_plot,
    global_breaks = global_breaks,
    stable_models = stable_models,
    volatile_models = volatile_models
  )

  if (is.null(ccc_df) || nrow(ccc_df) == 0) return(NULL)

  p <- ggplot(
    ccc_df,
    aes(x = DifficultyBin, y = Accuracy, group = Model, color = Group, linetype = Group)
  ) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = c("Stable" = "forestgreen", "Volatile" = "firebrick", "Other" = "grey60")) +
    scale_linetype_manual(values = c("Stable" = "solid", "Volatile" = "dashed", "Other" = "solid")) +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("Classifier Characteristic Curves —", dataset, regime),
      subtitle = "Stable vs Volatile Models",
      x = "Item difficulty bin (quintiles)",
      y = "Mean Accuracy"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  out_file <- file.path(out_root, paste0("CCC_", dataset, "_", regime, ".png"))
  ggsave(out_file, p, width = 8, height = 6, dpi = 300)
  p
}

imagenet_ip_zero <- extract_item_params("ImageNet", "zero_shot", fits_root)
global_breaks <- NULL
if (!is.null(imagenet_ip_zero) && "b_rank" %in% names(imagenet_ip_zero)) {
  global_breaks <- unique(quantile(imagenet_ip_zero$b_rank, probs = seq(0, 1, 0.2), na.rm = TRUE))
}

# choose stable/volatile sets
stable_zero_models <- if (!is.null(top_zero_acc)) tolower(head(top_zero_acc$stable$Model, 3)) else character(0)
volatile_zero_models <- if (!is.null(top_zero_acc)) tolower(head(top_zero_acc$volatile$Model, 3)) else character(0)

stable_train_models <- if (!is.null(top_train_theta)) tolower(head(top_train_theta$stable$Model, 3)) else character(0)
volatile_train_models <- if (!is.null(top_train_theta)) tolower(head(top_train_theta$volatile$Model, 3)) else character(0)

zero_shot_datasets <- c("ImageNet", "ImageNet-C", "Sketch", "ImageNet-Sketch")
for (ds in zero_shot_datasets) {
  plot_ccc(
    dataset = ds,
    regime = "zero_shot",
    models_to_plot = unique(c(stable_zero_models, volatile_zero_models)),
    global_breaks = global_breaks,
    stable_models = stable_zero_models,
    volatile_models = volatile_zero_models
  )
}

trained_like_runs <- tibble(
  dataset = c("ImageNet-C", "Sketch", "ImageNet-Sketch", "CIFAR100"),
  regime  = c("trained", "trained", "trained", "head_only")
)

for (i in seq_len(nrow(trained_like_runs))) {
  plot_ccc(
    dataset = trained_like_runs$dataset[i],
    regime = trained_like_runs$regime[i],
    models_to_plot = unique(c(stable_train_models, volatile_train_models)),
    global_breaks = global_breaks,
    stable_models = stable_train_models,
    volatile_models = volatile_train_models
  )
}

# ------------------------------------------------
# 4) Optional difficulty–accuracy correlation summary
# ------------------------------------------------
# Build from repo outputs if possible, instead of relying on an old precomputed file.

compute_b_acc_correlations <- function(dataset, regime) {
  ip <- extract_item_params(dataset, regime, fits_root)
  if (is.null(ip) || !"b" %in% names(ip)) return(NULL)

  model_dirs <- list.dirs(file.path(pred_root, dataset, regime), recursive = FALSE, full.names = TRUE)
  if (length(model_dirs) == 0) return(NULL)

  out <- list()

  for (md in model_dirs) {
    model <- basename(md)
    pred_df <- load_prediction_long(dataset, regime, model, pred_root)
    if (is.null(pred_df)) next

    merged <- pred_df %>%
      left_join(ip %>% select(Item, b), by = "Item") %>%
      filter(!is.na(b), !is.na(correct))

    if (nrow(merged) < 10) next

    rho <- suppressWarnings(cor(merged$b, merged$correct, method = "spearman", use = "pairwise.complete.obs"))
    out[[model]] <- tibble(
      Dataset = dataset,
      Regime = regime,
      Model = tolower(model),
      rho = rho
    )
  }

  bind_rows(out)
}

b_acc_all <- bind_rows(
  lapply(zero_shot_datasets, function(ds) compute_b_acc_correlations(ds, "zero_shot")),
  lapply(seq_len(nrow(trained_like_runs)), function(i) compute_b_acc_correlations(trained_like_runs$dataset[i], trained_like_runs$regime[i]))
)

if (!is.null(b_acc_all) && nrow(b_acc_all) > 0) {
  write_csv(b_acc_all, file.path(out_root, "b_acc_correlations_per_model.csv"))

  summary_corr_all <- bind_rows(
    if (!is.null(top_zero_acc)) {
      b_acc_all %>%
        filter(Regime == "zero_shot") %>%
        mutate(
          Group = case_when(
            Model %in% tolower(head(top_zero_acc$stable$Model, 10)) ~ "Stable",
            Model %in% tolower(head(top_zero_acc$volatile$Model, 10)) ~ "Volatile",
            TRUE ~ "Other"
          ),
          Mode = "zero_shot"
        )
    },
    if (!is.null(top_train_theta)) {
      b_acc_all %>%
        filter(Regime %in% c("trained", "head_only")) %>%
        mutate(
          Group = case_when(
            Model %in% tolower(head(top_train_theta$stable$Model, 10)) ~ "Stable",
            Model %in% tolower(head(top_train_theta$volatile$Model, 10)) ~ "Volatile",
            TRUE ~ "Other"
          ),
          Mode = "trained_like"
        )
    }
  )

  if (nrow(summary_corr_all) > 0) {
    write_csv(summary_corr_all, file.path(out_root, "b_acc_grouped_summary.csv"))

    p <- ggplot(summary_corr_all, aes(x = rho, fill = Group)) +
      geom_density(alpha = 0.4) +
      geom_vline(xintercept = 0, linetype = "dotted") +
      scale_fill_manual(values = c("Stable" = "forestgreen", "Volatile" = "firebrick", "Other" = "grey70")) +
      theme_minimal(base_size = 14) +
      labs(
        title = "Distribution of Spearman ρ(b, accuracy)",
        x = "Spearman correlation ρ(b, accuracy)",
        y = "Density"
      )

    ggsave(file.path(out_root, "b_acc_rho_distribution.png"), p, width = 8, height = 6, dpi = 300)
  }
}

# ------------------------------------------------
# 5) Output index
# ------------------------------------------------
output_index <- tibble(
  files = list.files(out_root, full.names = FALSE)
)

write_csv(output_index, file.path(out_root, "Visualization_Outputs.csv"))

cat("Saved visualization outputs to:\n", out_root, "\n", sep = "")
cat("Done.\n")
