# ============================================================
# difficulty_alignment.R
# Compare difficulty structure across datasets and modes
# Repo-safe version for difficulty alignment across datasets
# ============================================================

rm(list = ls())

library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(yaml)

# -----------------------------
# Load configuration
# -----------------------------
config <- yaml.load_file("config/config.yml")

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."
input_root <- file.path(repo_root, config$paths$response_matrices)  # Path for response matrices
out_root <- file.path(repo_root, config$paths$analysis, "irt", "difficulty_alignment")  # Output for difficulty alignment

# Create output directory if not exist
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# Define datasets and modes
datasets <- c("Sketch", "ImageNet-C", "CIFAR100")
modes_to_check <- c("zeroshot", "trained")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

# Function to load item parameters (b-values)
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
  if (!"b" %in% names(ip)) {
    stop("❌ No difficulty (b) column found in ", dataset, " (", mode, ")")
  }

  ip$Dataset <- dataset
  ip$Mode    <- mode
  return(ip)
}

# ---------------------------------------------------------
# Main function for difficulty alignment
# ---------------------------------------------------------
align_difficulties <- function(dataset) {
  message("=== Difficulty Alignment for: ", dataset, " ===")
  
  ip_zs <- load_item_params(dataset, "zeroshot")
  ip_tr <- load_item_params(dataset, "trained")
  
  if (is.null(ip_zs) || is.null(ip_tr)) {
    message("⚠️ Missing parameters for ", dataset, " — skipping difficulty alignment.")
    return(NULL)
  }
  
  # Calculate the mean and SD of b-values for both models
  mean_b_zs <- mean(ip_zs$b, na.rm = TRUE)
  sd_b_zs   <- sd(ip_zs$b,   na.rm = TRUE)
  
  mean_b_tr <- mean(ip_tr$b, na.rm = TRUE)
  sd_b_tr   <- sd(ip_tr$b,   na.rm = TRUE)
  
  # Scale the target model (trained) b-values to align with zeroshot
  A <- sd_b_zs / sd_b_tr
  B <- mean_b_zs - A * mean_b_tr
  
  b_linked <- A * ip_tr$b + B
  
  # Save results
  ip_tr$b_linked <- b_linked
  
  write.csv(
    ip_tr,
    file = file.path(out_root, paste0(dataset, "_difficulty_aligned.csv")),
    row.names = FALSE
  )
  
  message("🔗 Difficulty alignment saved to: ", file.path(out_root, paste0(dataset, "_difficulty_aligned.csv")))
}

# ---------------------------------------------------------
# Part A: Class-level Difficulty Alignment
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
    
    # Extract class-level difficulty
    class_zs <- get_class_difficulty(ip_zs)
    class_tr <- get_class_difficulty(ip_tr)
    
    merged_class <- inner_join(class_zs, class_tr, by = "Class")
    
    # Plot class-level alignment
    p <- ggplot(merged_class, aes(x = mean_b.x, y = mean_b.y)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "firebrick") +
      theme_minimal() +
      labs(
        title = paste0(ds, " Class-level Difficulty Alignment"),
        x = paste0(ds, " (zeroshot) - class mean difficulty b"),
        y = paste0(ds, " (trained) - class mean difficulty b")
      )
    
    # Save the plot
    out_file <- file.path(class_plot_dir, paste0("class_difficulty_alignment_", ds, ".png"))
    ggsave(out_file, p, width = 7, height = 5, dpi = 300)
  }
}

# ---------------------------------------------------------
# Part B: Superclass-level Difficulty Alignment
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
    
    # Extract superclass-level difficulty
    super_zs <- get_superclass_difficulty(ip_zs, ds)
    super_tr <- get_superclass_difficulty(ip_tr, ds)
    
    merged_super <- inner_join(super_zs, super_tr, by = "Superclass")
    
    # Plot superclass-level alignment
    p <- ggplot(merged_super, aes(x = mean_b.x, y = mean_b.y)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "firebrick") +
      theme_minimal() +
      labs(
        title = paste0(ds, " Superclass-level Difficulty Alignment"),
        x = paste0(ds, " (zeroshot) - superclass mean difficulty b"),
        y = paste0(ds, " (trained) - superclass mean difficulty b")
      )
    
    # Save the plot
    out_file <- file.path(super_plot_dir, paste0("super_difficulty_alignment_", ds, ".png"))
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

cat("Difficulty alignment completed.\n")
