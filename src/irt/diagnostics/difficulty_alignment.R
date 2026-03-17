# ==========================================================
# 04_difficulty_structure_alignment.R
# Compare difficulty structure across datasets/modes
# - Class-level (ImageNet, Sketch, ImageNet-C)
# - Superclass-level (ImageNet, Sketch, ImageNet-C, CIFAR100)
#
# Reference scale: ImageNet (zeroshot)
# ==========================================================
rm(list = ls())
library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(readr)
library(tibble)
library(patchwork)  # for grid layouts

# ---------------------------
# Thesis-ready plot theme
# ---------------------------
theme_thesis <- function(base_size = 10, base_family = "serif") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title    = element_text(face = "bold"),
      axis.title    = element_text(size = base_size),
      axis.text     = element_text(size = base_size - 1),
      plot.subtitle = element_text(size = base_size - 1),
      plot.caption  = element_text(size = base_size - 2),
      legend.position = "none",
      plot.margin = margin(8, 10, 8, 10)
    )
}



# Display names (do NOT change the internal dataset keys used by load_item_params)
ds_display <- function(ds) {
  if (ds == "Sketch") return("ImageNet-Sketch")
  if (ds == "CIFAR100") return("CIFAR-100")
  return(ds)
}

md_display <- function(md) {
  if (md == "zeroshot") return("Zero-shot")
  if (md == "trained")  return("Trained")
  return(md)
}

# Strip heavy text for grid panels (keep axes/ticks)
as_grid_panel <- function(p) {
  p +
    labs(x = NULL, y = NULL, caption = NULL, subtitle = NULL) +
    theme(
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      axis.text = element_text(size = 9),
      axis.title = element_blank(),
      plot.margin = margin(4, 4, 4, 4)
    )
}

ols_label_pos <- function(x, y, frac = 0.90, offset_frac = 0.04) {
  ok <- is.finite(x) & is.finite(y)
  x <- x[ok]; y <- y[ok]
  if (length(x) < 2) return(list(x = NA_real_, y = NA_real_))
  
  fit <- lm(y ~ x)
  b <- coef(fit)[2]
  
  xr <- range(x, na.rm = TRUE)
  yr <- range(y, na.rm = TRUE)
  
  x0 <- xr[1] + frac * (xr[2] - xr[1])
  y0 <- as.numeric(predict(fit, newdata = data.frame(x = x0)))
  
  # scale the offset to panel ranges (so CIFAR and ImageNet-C behave similarly)
  dx <- offset_frac * (xr[2] - xr[1])
  dy <- offset_frac * (yr[2] - yr[1])
  
  # push label away from the line depending on slope sign
  # (positive slope -> up; negative slope -> down; flat -> up)
  sgn <- ifelse(is.na(b) || abs(b) < 1e-6, 1, sign(b))
  x_lab <- x0 + 0.10 * dx
  y_lab <- y0 + sgn * 0.35 * dy
  
  # clamp to stay inside panel
  x_lab <- max(min(x_lab, xr[2] - 0.02 * (xr[2] - xr[1])), xr[1] + 0.02 * (xr[2] - xr[1]))
  y_lab <- max(min(y_lab, yr[2] - 0.02 * (yr[2] - yr[1])), yr[1] + 0.02 * (yr[2] - yr[1]))
  
  list(x = x_lab, y = y_lab)
}

# -----------------------------
# Load configuration
# -----------------------------
config <- yaml.load_file("config/config.yml")

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."

source("src/irt/diagnostics/dif_tif_linking.R")

if (!exists("load_item_params")) stop("load_item_params() not found.")
if (!exists("extract_class"))   stop("extract_class() not found.")
if (!exists("map_df"))          stop("map_df not found.")
if (!exists("cifar_map"))       stop("cifar_map not found.")

irt_root    <- file.path(repo_root, config$paths$theta_estimates)
diff_root  <- file.path(repo_root, config$paths$analysis, "irt", "difficulty_alignment")

dir.create(diff_root, recursive = TRUE, showWarnings = FALSE)

class_plot_dir <- file.path(diff_root, "class_level_plots")
super_plot_dir <- file.path(diff_root, "superclass_level_plots")

dir.create(class_plot_dir, showWarnings = FALSE)
dir.create(super_plot_dir, showWarnings = FALSE)

datasets_all   <- c("ImageNet", "Sketch", "ImageNet-C", "CIFAR100")
modes_to_check <- c("zeroshot", "trained")

get_class_difficulty <- function(ip) {
  ip %>%
    mutate(Class = extract_class(Item)) %>%
    group_by(Class) %>%
    summarise(
      mean_b   = mean(b, na.rm = TRUE),
      n_items  = n(),
      .groups  = "drop"
    )
}

get_superclass_difficulty <- function(ip, dataset) {
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
      mean_b   = mean(b, na.rm = TRUE),
      n_items  = n(),
      .groups  = "drop"
    )
}

# 1) Reference structure
ip_ref <- load_item_params("ImageNet", "zeroshot")
if (is.null(ip_ref)) stop("Could not load ImageNet (zeroshot) item parameters.")

class_ref <- get_class_difficulty(ip_ref) %>% rename(mean_b_ref = mean_b)
super_ref <- get_superclass_difficulty(ip_ref, "ImageNet") %>% rename(mean_b_ref = mean_b)

# 2) CLASS-LEVEL alignment (unchanged)
datasets_class <- c("ImageNet", "Sketch", "ImageNet-C")
class_align_list <- list()

for (ds in datasets_class) {
  for (md in modes_to_check) {
    
    if (ds == "ImageNet" && md == "zeroshot") next
    
    cat("\n=======================================\n")
    cat("Class-level difficulty alignment:", ds, "(", md, ")\n")
    
    ip_tgt <- load_item_params(ds, md)
    if (is.null(ip_tgt)) {
      cat("  ⚠️ No item parameters for", ds, md, "- skipping.\n")
      next
    }
    
    class_tgt <- get_class_difficulty(ip_tgt) %>% rename(mean_b_tgt = mean_b)
    merged <- inner_join(class_ref, class_tgt, by = "Class")
    
    n_common <- nrow(merged)
    if (n_common < 5) {
      cat("  ⚠️ Only", n_common, "common classes; skipping correlation.\n")
      next
    }
    
    rho_s <- suppressWarnings(cor(merged$mean_b_ref, merged$mean_b_tgt, method = "spearman", use = "complete.obs"))
    rho_p <- suppressWarnings(cor(merged$mean_b_ref, merged$mean_b_tgt, method = "pearson",  use = "complete.obs"))
    
    class_align_list[[paste(ds, md, sep = "_")]] <- data.frame(
      Dataset      = ds,
      Mode         = md,
      Level        = "Class",
      n_common     = n_common,
      spearman_rho = rho_s,
      pearson_r    = rho_p
    )
    
    p <- ggplot(merged, aes(x = mean_b_ref, y = mean_b_tgt)) +
      geom_point(alpha = 0.5) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "firebrick") +
      theme_minimal() +
      labs(
        title = paste0("Class difficulty alignment: ImageNet (zeroshot) vs ", ds, " (", md, ")"),
        subtitle = sprintf("Common classes: %d | Spearman ρ = %.3f | Pearson r = %.3f", n_common, rho_s, rho_p),
        x = "ImageNet (zeroshot) – class mean difficulty b",
        y = paste0(ds, " (", md, ") – class mean difficulty b")
      )
    
    out_file <- file.path(class_plot_dir, paste0("class_difficulty_alignment_ImageNet_zs_vs_", ds, "_", md, ".png"))
    ggsave(out_file, p, width = 7, height = 5, dpi = 300)
  }
}

if (length(class_align_list) > 0) {
  fwrite(bind_rows(class_align_list), file = file.path(diff_root, "difficulty_alignment_class_level_summary.csv"))
}

# 3) SUPERCLASS-LEVEL alignment + collect for grid
super_align_list <- list()
datasets_super   <- datasets_all

super_grid_order <- c("ImageNet-C", "Sketch", "CIFAR100")
super_grid_plots <- list(zeroshot = list(), trained = list())

# We only *compute* global limits during the loop,
# and apply them later when building the grid (ensures consistency).
super_grid_limits <- list(x = NULL, y = NULL)

for (ds in datasets_super) {
  for (md in modes_to_check) {
    
    if (ds == "ImageNet" && md == "zeroshot") next
    
    cat("\n=======================================\n")
    cat("Superclass-level difficulty alignment:", ds, "(", md, ")\n")
    
    ip_tgt <- load_item_params(ds, md)
    if (is.null(ip_tgt)) {
      cat("  ⚠️ No item parameters for", ds, md, "- skipping.\n")
      next
    }
    
    super_tgt <- get_superclass_difficulty(ip_tgt, ds) %>% rename(mean_b_tgt = mean_b)
    merged <- inner_join(super_ref, super_tgt, by = "Superclass")
    
    n_common <- nrow(merged)
    if (n_common < 3) next
    
    rho_s <- suppressWarnings(cor(merged$mean_b_ref, merged$mean_b_tgt, method = "spearman", use = "complete.obs"))
    rho_p <- suppressWarnings(cor(merged$mean_b_ref, merged$mean_b_tgt, method = "pearson",  use = "complete.obs"))
    
    super_align_list[[paste(ds, md, sep = "_")]] <- data.frame(
      Dataset      = ds,
      Mode         = md,
      Level        = "Superclass",
      n_common     = n_common,
      spearman_rho = rho_s,
      pearson_r    = rho_p
    )
    
    # update global grid limits (final values used later)
    if (ds %in% super_grid_order) {
      super_grid_limits$x <- range(c(super_grid_limits$x, merged$mean_b_ref), na.rm = TRUE)
      super_grid_limits$y <- range(c(super_grid_limits$y, merged$mean_b_tgt), na.rm = TRUE)
    }
    lim <- max(abs(c(merged$mean_b_ref, merged$mean_b_tgt)), na.rm = TRUE)
    
    panel_title <- paste0(ds_display(ds), " (", md_display(md), ")")
    
    title_expr <- if (is.na(rho_s)) {
      bquote(bold(atop(.(panel_title), rho == NA)))
    } else {
      bquote(bold(atop(.(panel_title), rho == .(formatC(rho_s, format = "f", digits = 2)))))
    }
    
    p <- ggplot(merged, aes(x = mean_b_ref, y = mean_b_tgt)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dotted", linewidth = 0.7, color = "grey50") +
      geom_point(shape = 21, size = 2.4, stroke = 0.35, color = "black", fill = "skyblue2", alpha = 0.9) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.8, color = "black") +
      coord_equal(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
      theme_thesis() +
      labs(
        title = title_expr,
        subtitle = sprintf("n=%d", n_common),
        x = "ImageNet (Zero-shot) – superclass mean difficulty b",
        y = paste0(ds_display(ds), " – superclass mean difficulty b"),
        caption = "Dotted line: y = x (perfect agreement). Solid line: OLS fit."
      )
    
    
    
    # store *without* coord_cartesian; we apply global limits later for consistency
    if (ds %in% super_grid_order) {
      super_grid_plots[[md]][[ds]] <- as_grid_panel(p)
    }
    
    out_file <- file.path(super_plot_dir, paste0("super_difficulty_alignment_ImageNet_zs_vs_", ds_display(ds), "_", md, ".png"))
    ggsave(out_file, p, width = 7, height = 5, dpi = 400)
    
    out_file_pdf <- file.path(super_plot_dir, paste0("super_difficulty_alignment_ImageNet_zs_vs_", ds_display(ds), "_", md, ".pdf"))
    ggsave(out_file_pdf, p, width = 7, height = 5)
  }
}

if (length(super_align_list) > 0) {
  fwrite(bind_rows(super_align_list), file = file.path(diff_root, "difficulty_alignment_superclass_level_summary.csv"))
}

# 4) Build + save grid (2x3)
blank_plot <- function(label) {
  ggplot() + theme_void() +
    annotate("text", x = 0, y = 0, label = label, size = 4)
}


get_plot_or_blank <- function(mode, ds_key) {
  p <- super_grid_plots[[mode]][[ds_key]]
  if (is.null(p)) {
    return(blank_plot(paste0(ds_display(ds_key), "\n(", md_display(mode), ")\nmissing")))
  }
  # apply final global limits consistently across all panels
  p + coord_cartesian(xlim = super_grid_limits$x, ylim = super_grid_limits$y)
}

p_zs_1 <- get_plot_or_blank("zeroshot", "ImageNet-C")
p_zs_2 <- get_plot_or_blank("zeroshot", "Sketch")
p_zs_3 <- get_plot_or_blank("zeroshot", "CIFAR100")

p_tr_1 <- get_plot_or_blank("trained", "ImageNet-C")
p_tr_2 <- get_plot_or_blank("trained", "Sketch")
p_tr_3 <- get_plot_or_blank("trained", "CIFAR100")

super_grid <- (p_zs_1 | p_zs_2 | p_zs_3) /
  (p_tr_1 | p_tr_2 | p_tr_3) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Superclass difficulty alignment vs ImageNet (Zero-shot)",
    subtitle = "Top: Zero-shot   |   Bottom: Trained",
    caption = "x-axis: ImageNet (Zero-shot) superclass mean difficulty b   |   y-axis: Target dataset superclass mean difficulty b",
    theme = theme(
      plot.title = element_text(face = "bold"),
      plot.margin = margin(2, 2, 2, 2)
    )
  )

grid_png <- file.path(super_plot_dir, "superclass_alignment_grid_2x3.png")
ggsave(grid_png, super_grid, width = 12.5, height = 7.5, dpi = 400)

grid_png_pdf <- file.path(super_plot_dir, "superclass_alignment_grid_2x3.pdf")
ggsave(grid_png_pdf, super_grid, width = 12.5, height = 7.5)
cat("\nSaved superclass 2x3 grid to:\n  ", grid_png, "\n")
