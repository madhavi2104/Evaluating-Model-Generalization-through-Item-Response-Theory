# ============================================================
# discrimination_alignment.R
# Compare discrimination structure across datasets and modes
# Repo-safe version for discrimination alignment across datasets
# ============================================================

rm(list = ls())

library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(yaml)
library(patchwork)

# -----------------------------
# Load configuration
# -----------------------------
config <- yaml.load_file("config/config.yml")

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."
input_root <- file.path(repo_root, config$paths$response_matrices)  # Path for response matrices
out_root <- file.path(repo_root, config$paths$analysis, "irt", "discrimination_alignment")  # Output for discrimination alignment

# Create output directory if not exist
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# Define datasets and modes
datasets <- c("Sketch", "ImageNet-C", "CIFAR100")
modes_to_check <- c("zeroshot", "trained")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

# Function to load item parameters (a-values)
load_item_params <- function(dataset, mode) {
  base_dir <- file.path(config$paths$theta_estimates, dataset, mode)
  best_dirs <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  best_dirs <- best_dirs[grepl("best_", basename(best_dirs))]
  
  if (length(best_dirs) == 0) {
    message("⚠️ No best_* folder found for ", dataset, " (", mode, ")")
    return(NULL)
  }
  
  best_dir <- best_dirs[1]
  ip_file  <- file.path(best_dir, "ItemParameters.rds")
  if (!file.exists(ip_file)) {
    message("⚠️ Missing ItemParameters.rds in ", best_dir)
    return(NULL)
  }
  
  ip <- readRDS(ip_file) %>%
    as.data.frame() %>%
    rownames_to_column("Item")

  # Ensure columns are as expected
  if (!"a" %in% names(ip)) {
    stop("❌ No discrimination (a) column found in ", dataset, " (", mode, ")")
  }

  ip$Dataset <- dataset
  ip$Mode    <- mode
  return(ip)
}

# Function for class-level discrimination
get_class_discrimination <- function(ip) {
  ip %>%
    mutate(Class = extract_class(Item)) %>%
    group_by(Class) %>%
    summarise(
      mean_a  = mean(a, na.rm = TRUE),
      n_items = n(),
      .groups = "drop"
    )
}

# Function for superclass-level discrimination
get_superclass_discrimination <- function(ip, dataset) {
  df <- ip %>%
    mutate(Class = extract_class(Item))
  
  df <- df %>%
    left_join(map_df, by = c("Class" = "imagenet_id")) %>%
    mutate(Superclass = cifar_superclass)
  
  if (dataset == "CIFAR100") {
    df <- df %>%
      left_join(cifar_map, by = c("Class" = "Fine")) %>%
      mutate(Superclass = ifelse(!is.na(Superclass.y), Superclass.y, Superclass.x)) %>%
      select(-Superclass.x, -Superclass.y)
  }
  
  df %>%
    filter(!is.na(Superclass)) %>%
    mutate(Superclass = tolower(trimws(Superclass))) %>%
    group_by(Superclass) %>%
    summarise(
      mean_a  = mean(a, na.rm = TRUE),
      n_items = n(),
      .groups = "drop"
    )
}

# ---------------------------------------------------------
# Part A: Class-level Discrimination Alignment
# ---------------------------------------------------------
class_level_alignment <- function() {
  class_plot_dir <- file.path(out_root, "class_level_plots")
  dir.create(class_plot_dir, showWarnings = FALSE)

  for (ds in datasets) {
    ip_zs <- load_item_params(ds, "zeroshot")
    ip_tr <- load_item_params(ds, "trained")
    
    if (is.null(ip_zs) || is.null(ip_tr)) {
      message("⚠️ Missing item parameters for ", ds, " — skipping class-level alignment.")
      next
    }
    
    # Extract class-level discrimination
    class_zs <- get_class_discrimination(ip_zs)
    class_tr <- get_class_discrimination(ip_tr)
    
    merged_class <- inner_join(class_zs, class_tr, by = "Class")
    
    # Plot class-level alignment
    p <- ggplot(merged_class, aes(x = mean_a.x, y = mean_a.y)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "firebrick") +
      theme_minimal() +
      labs(
        title = paste0(ds, " Class-level Discrimination Alignment"),
        x = paste0(ds, " (zeroshot) - class mean discrimination a"),
        y = paste0(ds, " (trained) - class mean discrimination a")
      )
    
    # Save the plot
    out_file <- file.path(class_plot_dir, paste0("class_discrimination_alignment_", ds, ".png"))
    ggsave(out_file, p, width = 7, height = 5, dpi = 300)
  }
}

# ---------------------------------------------------------
# Part B: Superclass-level Discrimination Alignment
# ---------------------------------------------------------
superclass_level_alignment <- function() {
  super_plot_dir <- file.path(out_root, "superclass_level_plots")
  dir.create(super_plot_dir, showWarnings = FALSE)
  
  for (ds in datasets) {
    ip_zs <- load_item_params(ds, "zeroshot")
    ip_tr <- load_item_params(ds, "trained")
    
    if (is.null(ip_zs) || is.null(ip_tr)) {
      message("⚠️ Missing item parameters for ", ds, " — skipping superclass-level alignment.")
      next
    }
    
    # Extract superclass-level discrimination
    super_zs <- get_superclass_discrimination(ip_zs, ds)
    super_tr <- get_superclass_discrimination(ip_tr, ds)
    
    merged_super <- inner_join(super_zs, super_tr, by = "Superclass")
    
    # Plot superclass-level alignment
    p <- ggplot(merged_super, aes(x = mean_a.x, y = mean_a.y)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "firebrick") +
      theme_minimal() +
      labs(
        title = paste0(ds, " Superclass-level Discrimination Alignment"),
        x = paste0(ds, " (zeroshot) - superclass mean discrimination a"),
        y = paste0(ds, " (trained) - superclass mean discrimination a")
      )
    
    # Save the plot
    out_file <- file.path(super_plot_dir, paste0("super_discrimination_alignment_", ds, ".png"))
    ggsave(out_file, p, width = 7, height = 5, dpi = 300)
  }
}

# ---------------------------------------------------------
# Part C: Global Comparison Plot (1x3 grid)
# ---------------------------------------------------------
global_comparison_plot <- function() {
  # Prepare 1x3 grid plot comparing all datasets (zeroshot vs trained)
  datasets_for_comparison <- c("ImageNet-C", "ImageNet-Sketch", "CIFAR100")
  
  grid_plot <- lapply(datasets_for_comparison, function(ds) {
    p_zs <- ggplot() + geom_point()  # Placeholder for zeroshot plot
    p_tr <- ggplot() + geom_point()  # Placeholder for trained plot
    
    # Combine into a 1x3 grid using `patchwork` or custom grid
    p_combined <- p_zs | p_tr
    return(p_combined)
  })
  
  # Save final grid plot
  grid_plot_file <- file.path(out_root, "global_comparison_grid.png")
  ggsave(grid_plot_file, grid_plot, width = 12, height = 7, dpi = 300)
}

# ---------------------------------------------------------
# Running all parts
# ---------------------------------------------------------
class_level_alignment()
superclass_level_alignment()
global_comparison_plot()

cat("Discrimination alignment completed.\n")
