library(ggplot2)
library(dplyr)
library(tidyr)

# --- Config ---
input_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/analysis"
out_root   <- file.path(input_root, "plots_H1")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)


## 2PL Item Characteristic Curve (ICC) example -----------------------------

# Load packages
library(ggplot2)

# 2PL parameters (you can change these if you like)
a <- 2      # discrimination
b <- 3      # difficulty (location)

# Theta grid
theta <- seq(-2, 7, length.out = 500)

# 2PL curve: P(Y = 1 | theta)
p <- 1 / (1 + exp(-a * (theta - b)))

# Value and slope at theta = b
p_b    <- 1 / (1 + exp(-a * (b - b)))   # should be 0.5
slope  <- a / 4                         # derivative of 2PL at theta = b

# Tangent line around theta = b (for visualising discrimination)
theta_tan <- seq(b - 2, b + 2, length.out = 200)
p_tan     <- p_b + slope * (theta_tan - b)

# Assemble data frames
df_curve <- data.frame(theta = theta,    p = p)
df_tan   <- data.frame(theta = theta_tan, p = p_tan)
df_point <- data.frame(theta = b, p = p_b)

# Plot ---------------------------------------------------------------------
p_2pl <- ggplot(df_curve, aes(x = theta, y = p)) +
  # ICC curve
  geom_line(size = 1) +
  
  # Tangent line at b (discrimination)
  geom_line(data = df_tan,
            linetype = "dashed",
            colour = "red") +
  
  # Vertical line at difficulty b
  geom_vline(xintercept = b,
             linetype = "dotted",
             colour = "darkgreen") +
  
  # Point at (b, 0.5)
  geom_point(data = df_point,
             size = 2) +
  
  # Axis labels
  labs(x = expression(theta),
       y = "Probability of Correct Response") +
  
  # Annotations for a and b (optional, delete if you want it cleaner)
  annotate("text",
           x = b + 0.2, y = 0.52,
           label = "b",
           hjust = 0) +
  annotate("text",
           x = b + 1.8, y = p_b + slope * 1.8,
           label = "slope == a/4",
           hjust = 0,
           parse = TRUE,
           colour = "red") +
  
  # Clean, scientific theme
  coord_cartesian(ylim = c(0, 1)) +
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank()
  )

# Print to RStudio plot pane
print(p_2pl)

# Save the plot
ggsave(
  filename = file.path(out_root, "2PL_ICC_example.pdf"),
  plot = p_2pl,
  width = 5,
  height = 4,
  units = "in"
)


# ------------------------------------------------
# ================ ImageNot Ranks ================
# ------------------------------------------------
# ImageNot-6 models
imagenot_models <- c("alexnet",
                     "vgg19",
                     "resnet152",
                     "densenet161",
                     "efficientnetv2_l",
                     "convnext_large")

# --- Helper: compute relative ranks (1–6) ---
compute_relative_ranks <- function(df) {
  df_sub <- df %>% filter(Model %in% imagenot_models)
  # re-rank within each dataset
  df_re_ranked <- df_sub
  for (col in colnames(df_sub)[-1]) {
    df_re_ranked[[col]] <- rank(df_sub[[col]], ties.method = "first")
  }
  df_re_ranked
}

# --- Plot slope chart ---
make_slope_plot <- function(df, title_prefix, out_file) {
  df_long <- df %>%
    pivot_longer(-Model, names_to = "Dataset", values_to = "Rank") %>%
    mutate(
      Dataset = case_when(
        grepl("imagenet.c", Dataset, ignore.case = TRUE) ~ "ImageNet-C",
        grepl("imagenet[-_]?sketch|sketch", Dataset, ignore.case = TRUE) ~ "ImageNet-Sketch",
        grepl("cifar", Dataset, ignore.case = TRUE) ~ "CIFAR100",
        grepl("imagenet", Dataset, ignore.case = TRUE) ~ "ImageNet",
        TRUE ~ Dataset
      ),
      Dataset = factor(Dataset,
                       levels = c("ImageNet", "ImageNet-C", "ImageNet-Sketch", "CIFAR100"))
    )
  
  p <- ggplot(df_long, aes(x = Dataset, y = Rank, group = Model, color = Model)) +
    geom_line(size = 1.2) +
    geom_point(size = 8, shape = 21, fill = "white", stroke = 1.2) +
    geom_text(aes(label = Rank), color = "black", size = 3) +
    scale_y_reverse(breaks = 1:6, limits = c(6, 1)) +
    theme_minimal(base_size = 14) +
    labs(
      title = paste0(title_prefix, " (ImageNot-6 Models)"),
      y = "Relative Rank (1 = best within 6)",
      x = "Dataset"
    ) +
    theme(
      legend.position = "right",
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  print(p)
  ggsave(out_file, plot = p, width = 8, height = 6, dpi = 300)
  cat("✅ Saved plot:", out_file, "\n")
}


# === Run for each case ===

# 1. Mixed accuracy
df_mix_acc <- read.csv(file.path(input_root, "mix_trained_zeroshot_accuracy_rank_wide.csv"))
df_mix_acc_rel <- compute_relative_ranks(df_mix_acc)
write.csv(df_mix_acc_rel, file.path(input_root, "imagenot_mix_accuracy_ranks.csv"), row.names = FALSE)
make_slope_plot(df_mix_acc_rel, "Trained Accuracy Ranks", file.path(out_root, "slope_mix_accuracy.png"))

# 2. Mixed theta
df_mix_theta <- read.csv(file.path(input_root, "mix_trained_zeroshot_theta_rank_wide.csv"))
df_mix_theta_rel <- compute_relative_ranks(df_mix_theta)
write.csv(df_mix_theta_rel, file.path(input_root, "imagenot_mix_theta_ranks.csv"), row.names = FALSE)
make_slope_plot(df_mix_theta_rel, "Trained Theta Ranks", file.path(out_root, "slope_mix_theta.png"))

# 3. Zeroshot accuracy
df_zero_acc <- read.csv(file.path(input_root, "all_zeroshot_accuracy_rank_wide.csv"))
df_zero_acc_rel <- compute_relative_ranks(df_zero_acc)
write.csv(df_zero_acc_rel, file.path(input_root, "imagenot_zeroshot_accuracy_ranks.csv"), row.names = FALSE)
make_slope_plot(df_zero_acc_rel, "Zeroshot Accuracy Ranks", file.path(out_root, "slope_zeroshot_accuracy.png"))

# 4. Zeroshot theta
df_zero_theta <- read.csv(file.path(input_root, "all_zeroshot_theta_rank_wide.csv"))
df_zero_theta_rel <- compute_relative_ranks(df_zero_theta)
write.csv(df_zero_theta_rel, file.path(input_root, "imagenot_zeroshot_theta_ranks.csv"), row.names = FALSE)
make_slope_plot(df_zero_theta_rel, "Zeroshot Theta Ranks", file.path(out_root, "slope_zeroshot_theta.png"))



# -------------------------------------------------------------------
# =================== Scatter plots (rho) ===========================
# -------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(patchwork)
library(grid)
library(cowplot)


# --- Config ---
input_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/analysis"
out_root   <- file.path(input_root, "plots_H1")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# --- Helper: enforce numeric ---
make_numeric <- function(df) {
  for (col in colnames(df)[-1]) {
    df[[col]] <- as.numeric(df[[col]])
  }
  df
}

# --- Load data ---
acc_zero   <- make_numeric(read.csv(file.path(input_root, "all_zeroshot_accuracy_wide.csv")))
theta_zero <- make_numeric(read.csv(file.path(input_root, "all_zeroshot_theta_wide.csv")))
acc_mix    <- make_numeric(read.csv(file.path(input_root, "mix_trained_zeroshot_accuracy_wide.csv")))
theta_mix  <- make_numeric(read.csv(file.path(input_root, "mix_trained_zeroshot_theta_wide.csv")))

# --- Mapping: zeroshot column → mixed column ---
ds_map <- c(
  "CIFAR100_zeroshot"   = "CIFAR100_trained",
  "Sketch_zeroshot"     = "Sketch_trained",
  "ImageNet.C_zeroshot" = "ImageNet.C_trained"
)

# --- Helper: scatter plot with identity + rho ---
make_scatter <- function(x, y, x_label, y_label, rho, title) {
  df <- data.frame(X = x, Y = y)
  ggplot(df, aes(X, Y)) +
    geom_point(color = "steelblue", size = 2) +
    geom_smooth(method = "lm", se = FALSE,
                linetype = "dashed", color = "red") +
    theme_minimal(base_size = 12) +
    labs(x = x_label, y = y_label,
         title = title,
         subtitle = paste0("Spearman ρ = ", round(rho, 3)))
}

# --- Wrapper: make grid for one hypothesis ---
make_hypothesis_plot <- function(hypothesis, out_file) {
  plots_zero <- list()
  plots_mix  <- list()
  
  for (ds in names(ds_map)) {
    ds_mix <- ds_map[[ds]]
    
    if (hypothesis == "H1.1") {
      rho0 <- cor(acc_zero$ImageNet_zeroshot, acc_zero[[ds]], method="spearman")
      rhoM <- cor(acc_mix$ImageNet_zeroshot, acc_mix[[ds_mix]], method="spearman")
      
      plots_zero[[ds]] <- make_scatter(acc_zero$ImageNet_zeroshot, acc_zero[[ds]],
                                       "ImageNet acc", paste0(ds, " acc"), rho0, "")
      plots_mix[[ds]]  <- make_scatter(acc_mix$ImageNet_zeroshot, acc_mix[[ds_mix]],
                                       "ImageNet acc", paste0(ds_mix, " acc"), rhoM, "")
    }
    
    if (hypothesis == "H1.2") {
      rho0 <- cor(theta_zero[[ds]], acc_zero[[ds]], method="spearman")
      rhoM <- cor(theta_mix[[ds_mix]], acc_mix[[ds_mix]], method="spearman")
      
      plots_zero[[ds]] <- make_scatter(theta_zero[[ds]], acc_zero[[ds]],
                                       paste0(ds, " θ"), paste0(ds, " acc"), rho0, "")
      plots_mix[[ds]]  <- make_scatter(theta_mix[[ds_mix]], acc_mix[[ds_mix]],
                                       paste0(ds_mix, " θ"), paste0(ds_mix, " acc"), rhoM, "")
    }
    
    if (hypothesis == "H1.3") {
      rho0 <- cor(theta_zero$ImageNet_zeroshot, theta_zero[[ds]], method="spearman")
      rhoM <- cor(theta_mix$ImageNet_zeroshot, theta_mix[[ds_mix]], method="spearman")
      
      plots_zero[[ds]] <- make_scatter(theta_zero$ImageNet_zeroshot, theta_zero[[ds]],
                                       "ImageNet θ", paste0(ds, " θ"), rho0, "")
      plots_mix[[ds]]  <- make_scatter(theta_mix$ImageNet_zeroshot, theta_mix[[ds_mix]],
                                       "ImageNet θ", paste0(ds_mix, " θ"), rhoM, "")
    }
    
    if (hypothesis == "H1.4") {
      rho0 <- cor(acc_zero$ImageNet_zeroshot, theta_zero[[ds]], method="spearman")
      rhoM <- cor(acc_mix$ImageNet_zeroshot, theta_mix[[ds_mix]], method="spearman")
      
      plots_zero[[ds]] <- make_scatter(acc_zero$ImageNet_zeroshot, theta_zero[[ds]],
                                       "ImageNet acc", paste0(ds, " θ"), rho0, "")
      plots_mix[[ds]]  <- make_scatter(acc_mix$ImageNet_zeroshot, theta_mix[[ds_mix]],
                                       "ImageNet acc", paste0(ds_mix, " θ"), rhoM, "")
    }
    
    if (hypothesis == "H1.5") {
      rho0 <- cor(theta_zero$ImageNet_zeroshot, acc_zero[[ds]], method="spearman")
      rhoM <- cor(theta_mix$ImageNet_zeroshot, acc_mix[[ds_mix]], method="spearman")
      
      plots_zero[[ds]] <- make_scatter(theta_zero$ImageNet_zeroshot, acc_zero[[ds]],
                                       "ImageNet θ", paste0(ds, " acc"), rho0, "")
      plots_mix[[ds]]  <- make_scatter(theta_mix$ImageNet_zeroshot, acc_mix[[ds_mix]],
                                       "ImageNet θ", paste0(ds_mix, " acc"), rhoM, "")
    }
  }
  
  # Build rows (no titles at all)
  row_zero <- wrap_plots(plots_zero, ncol = 3)
  row_mix  <- wrap_plots(plots_mix,  ncol = 3)
  
  # Add one shared label per row
  row_zero_labeled <- plot_grid(NULL, row_zero, ncol = 1, rel_heights = c(0.05, 1))
  row_mix_labeled  <- plot_grid(NULL, row_mix,  ncol = 1, rel_heights = c(0.05, 1))
  
  # Actually add text labels as grobs
  row_zero_final <- plot_grid(ggdraw() + draw_label("Zeroshot", fontface = "bold", size = 16),
                              row_zero,
                              ncol = 1, rel_heights = c(0.1, 1))
  row_mix_final  <- plot_grid(ggdraw() + draw_label("Mixed", fontface = "bold", size = 16),
                              row_mix,
                              ncol = 1, rel_heights = c(0.1, 1))
  
  # Stack them
  grid <- plot_grid(row_zero_final, row_mix_final, ncol = 1)
  
  
  print(grid)
  ggsave(out_file, grid, width=15, height=10, dpi=300)
  cat("✅ Saved:", out_file, "\n")
}

# --- Run all hypotheses ---
make_hypothesis_plot("H1.1", file.path(out_root, "H1.1_grid.png"))
make_hypothesis_plot("H1.2", file.path(out_root, "H1.2_grid.png"))
make_hypothesis_plot("H1.3", file.path(out_root, "H1.3_grid.png"))
make_hypothesis_plot("H1.4", file.path(out_root, "H1.4_grid.png"))
make_hypothesis_plot("H1.5", file.path(out_root, "H1.5_grid.png"))


# ------------------------------------------------
# ========== Stable & Volatile Model Eval =========
# ------------------------------------------------

# --- Helper: compute rank variance across datasets ---
compute_rank_variance <- function(df, mode_label) {
  df_long <- df %>% pivot_longer(-Model, names_to="Dataset", values_to="Rank")
  df_var  <- df_long %>%
    group_by(Model) %>%
    summarise(Var = var(Rank, na.rm=TRUE),
              Range = max(Rank, na.rm=TRUE) - min(Rank, na.rm=TRUE)) %>%
    arrange(desc(Var))
  df_var$Mode <- mode_label
  df_var
}

# --- Load rank data (wide format) ---
rank_zero_acc   <- read.csv(file.path(input_root, "all_zeroshot_accuracy_rank_wide.csv"))
rank_zero_theta <- read.csv(file.path(input_root, "all_zeroshot_theta_rank_wide.csv"))
rank_mix_acc    <- read.csv(file.path(input_root, "mix_trained_zeroshot_accuracy_rank_wide.csv"))
rank_mix_theta  <- read.csv(file.path(input_root, "mix_trained_zeroshot_theta_rank_wide.csv"))

# --- Compute volatility tables ---
var_zero_acc   <- compute_rank_variance(rank_zero_acc,   "Zeroshot Accuracy")
var_zero_theta <- compute_rank_variance(rank_zero_theta, "Zeroshot Theta")
var_mix_acc    <- compute_rank_variance(rank_mix_acc,    "Trained Accuracy")
var_mix_theta  <- compute_rank_variance(rank_mix_theta,  "Trained Theta")

# --- Combine all ---
var_all <- bind_rows(var_zero_acc, var_zero_theta, var_mix_acc, var_mix_theta)
write.csv(var_all, file.path(input_root, "rank_volatility_summary.csv"), row.names=FALSE)

# --- Select top 10 stable & volatile per mode ---
get_top_models <- function(var_df, top_n=10) {
  stable   <- var_df %>% arrange(Var) %>% head(top_n)
  volatile <- var_df %>% arrange(desc(Var)) %>% head(top_n)
  list(stable=stable, volatile=volatile)
}

top_zero_acc   <- get_top_models(var_zero_acc)
top_zero_theta <- get_top_models(var_zero_theta)
top_mix_acc    <- get_top_models(var_mix_acc)
top_mix_theta  <- get_top_models(var_mix_theta)

# --- Function: slope plots for top-10 sets ---
make_top_slope <- function(df, top_models, title, out_file) {
  df_sub <- df %>% filter(Model %in% top_models$Model)
  df_long <- df_sub %>%
    pivot_longer(-Model, names_to="Dataset", values_to="Rank")
  
  p <- ggplot(df_long, aes(x=Dataset, y=Rank, group=Model, color=Model)) +
    geom_line(size=1.1) +
    geom_point(size=3, shape=21, fill="white", stroke=1.1) +
    scale_y_reverse() +
    theme_minimal(base_size=12) +
    labs(title=title, y="Rank (lower = better)") +
    theme(axis.text.x=element_text(angle=45, hjust=1),
          legend.position="right")
  print(p)
  ggsave(out_file, plot=p, width=9, height=6, dpi=300)
  cat("✅ Saved:", out_file, "\n")
}

# --- Make slope plots ---
make_top_slope(rank_zero_acc,   top_zero_acc$stable,   "Top-10 Stable Models (Zeroshot Accuracy)", file.path(out_root,"slope_stable_zero_acc.png"))
make_top_slope(rank_zero_acc,   top_zero_acc$volatile, "Top-10 Volatile Models (Zeroshot Accuracy)", file.path(out_root,"slope_volatile_zero_acc.png"))

make_top_slope(rank_zero_theta, top_zero_theta$stable,   "Top-10 Stable Models (Zeroshot Theta)", file.path(out_root,"slope_stable_zero_theta.png"))
make_top_slope(rank_zero_theta, top_zero_theta$volatile, "Top-10 Volatile Models (Zeroshot Theta)", file.path(out_root,"slope_volatile_zero_theta.png"))

make_top_slope(rank_mix_acc,    top_mix_acc$stable,   "Top-10 Stable Models (Trained Accuracy)", file.path(out_root,"slope_stable_mix_acc.png"))
make_top_slope(rank_mix_acc,    top_mix_acc$volatile, "Top-10 Volatile Models (Trained Accuracy)", file.path(out_root,"slope_volatile_mix_acc.png"))

make_top_slope(rank_mix_theta,  top_mix_theta$stable,   "Top-10 Stable Models (Trained Theta)", file.path(out_root,"slope_stable_mix_theta.png"))
make_top_slope(rank_mix_theta,  top_mix_theta$volatile, "Top-10 Volatile Models (Trained Theta)", file.path(out_root,"slope_volatile_mix_theta.png"))

# ------------------------------------------------
# ======== Classifier Characteristic Curves (CCC) ========
# ------------------------------------------------
# Visualizes how model accuracy varies across bins of IRT difficulty (b)
# ------------------------------------------------

library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)

# --- Config: paths ---
theta_root <- "E:/Thesis/IRTNet/output/Hypothesis_2/theta_est"
pred_root  <- "E:/Thesis/IRTNet/output/Hypothesis_2/predictions"
out_root   <- "E:/Thesis/IRTNet/output/Hypothesis_2/analysis/CCC"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# ------------------------------------------------
# Load item difficulties (auto-orientation check with CI)
# ------------------------------------------------
load_item_difficulties <- function(dataset, mode,
                                   pred_root = "E:/Thesis/IRTNet/output/Hypothesis_2/predictions",
                                   n_boot = 1000) {
  base_dir <- file.path(theta_root, dataset, mode)
  best_dir <- list.dirs(base_dir, recursive = FALSE, full.names = TRUE)
  best_dir <- best_dir[grepl("best_", basename(best_dir))][1]
  
  if (length(best_dir) == 0 || is.na(best_dir)) {
    message("⚠️ No best_* directory for ", dataset, " (", mode, ")")
    return(NULL)
  }
  
  ip_file <- file.path(best_dir, "ItemParameters.rds")
  if (!file.exists(ip_file)) {
    message("⚠️ Missing ItemParameters.rds for ", dataset, " (", mode, ")")
    return(NULL)
  }
  
  # --- Load item parameters ---
  ip <- readRDS(ip_file) %>% as.data.frame()
  ip$Item <- rownames(ip)
  if ("d" %in% names(ip)) names(ip)[names(ip) == "d"] <- "b"
  if (!"b" %in% names(ip)) stop("❌ No difficulty (b) column found for ", dataset, " (", mode, ")")
  
  # ------------------------------------------------
  # 🔍 Orientation sanity check with bootstrap CI
  # ------------------------------------------------
  pred_file <- file.path(pred_root, paste0(dataset, "_", mode, ".csv"))
  if (file.exists(pred_file)) {
    preds <- read.csv(pred_file, check.names = FALSE)
    if (!"Item" %in% names(preds)) names(preds)[1] <- "Item"
    
    item_acc <- preds %>%
      dplyr::select(-dplyr::matches("true.?label", ignore.case = TRUE)) %>%
      tidyr::pivot_longer(-Item, names_to = "Model", values_to = "Correct") %>%
      group_by(Item) %>%
      summarise(p_value = mean(Correct, na.rm = TRUE), .groups = "drop")
    
    merged <- left_join(ip, item_acc, by = "Item") %>%
      filter(!is.na(b) & !is.na(p_value))
    
    rho_obs <- suppressWarnings(cor(merged$b, merged$p_value, method = "spearman"))
    
    # Bootstrap CI for ρ(b,p)
    if (nrow(merged) > 10) {
      set.seed(123)
      boot_rhos <- replicate(n_boot, {
        idx <- sample(seq_len(nrow(merged)), replace = TRUE)
        suppressWarnings(cor(merged$b[idx], merged$p_value[idx],
                             method = "spearman", use = "complete.obs"))
      })
      ci <- quantile(boot_rhos, probs = c(0.025, 0.975), na.rm = TRUE)
    } else {
      ci <- c(NA, NA)
    }
    
    # Flip only if 95% CI entirely > 0 (clearly wrong orientation)
    if (!is.na(rho_obs) && !any(is.na(ci)) && ci[1] > 0) {
      ip$b <- -ip$b
      message(sprintf("🔄 Flipped difficulty orientation for %s (%s) [ρ=%.3f, 95%% CI (%.3f – %.3f)]",
                      dataset, mode, rho_obs, ci[1], ci[2]))
    } else if (!is.na(rho_obs) && !any(is.na(ci)) && ci[1] < 0 && ci[2] < 0) {
      message(sprintf("✅ Difficulty orientation OK for %s (%s) [ρ=%.3f, 95%% CI (%.3f – %.3f)]",
                      dataset, mode, rho_obs, ci[1], ci[2]))
    } else {
      message(sprintf("⚠️ Uncertain orientation for %s (%s) [ρ=%.3f, 95%% CI (%.3f – %.3f)] — not flipped",
                      dataset, mode, rho_obs, ci[1], ci[2]))
    }
  } else {
    message("⚠️ Skipping orientation check — no prediction file found for ", dataset, " (", mode, ")")
  }
  
  # ------------------------------------------------
  # 🧮 Normalize difficulty (AFTER flip)
  # ------------------------------------------------
  ip <- ip %>%
    mutate(
      b_rank = rank(b, ties.method = "average") / n(),  # normalized 0–1
      b_z = (b - mean(b, na.rm = TRUE)) / sd(b, na.rm = TRUE)
    )
  
  return(ip)
}


# --- Compute model accuracy per difficulty bin ---
compute_ccc <- function(pred_file, ip, model, bin_var = "b_rank", global_breaks = NULL) {
  preds <- read.csv(pred_file, check.names = FALSE)
  if (!"Item" %in% names(preds)) stop("Prediction file must include 'Item' column")
  
  df_long <- preds %>%
    pivot_longer(-c(Item, `True Label`), names_to = "Model", values_to = "Correct") %>%
    mutate(Correct = as.numeric(Correct))
  
  merged <- df_long %>%
    left_join(ip %>% select(Item, b, b_rank, b_z), by = "Item") %>%
    filter(Model == model)
  
  if (nrow(merged) == 0) return(NULL)
  
  # Use only non-missing standardized difficulty values
  b_col <- merged[[bin_var]]
  b_col <- b_col[!is.na(b_col)]
  
  # Define breaks (use global if available)
  if (is.null(global_breaks)) {
    qbreaks <- unique(quantile(b_col, probs = seq(0, 1, 0.2), na.rm = TRUE))
  } else {
    qbreaks <- unique(global_breaks)
  }
  # --- Apply binning safely (avoid NA bin for out-of-range values) ---
  merged <- merged %>%
    filter(!is.na(get(bin_var))) %>%
    mutate(
      DifficultyBin = cut(
        pmax(pmin(get(bin_var), max(qbreaks)), min(qbreaks)),  # clamp both ends
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
      Accuracy = mean(Correct, na.rm = TRUE),
      .groups = "drop"
    )
  
  acc_by_bin$Model <- model
  acc_by_bin
}

# --- Plot CCC with Stable vs Volatile labeling ---
plot_ccc <- function(dataset, mode, models, global_breaks = NULL, bin_var = "b_rank",
                     stable_models = NULL, volatile_models = NULL) {
  pred_file <- file.path(pred_root, paste0(dataset, "_", mode, ".csv"))
  ip <- load_item_difficulties(dataset, mode)
  if (is.null(ip)) return(NULL)
  
  ccc_all <- bind_rows(lapply(models, function(m)
    compute_ccc(pred_file, ip, m, bin_var, global_breaks)
  ))
  
  if (is.null(ccc_all) || nrow(ccc_all) == 0) {
    message("⚠️ No valid CCC data for ", dataset, " (", mode, ")")
    return(NULL)
  }
  
  ccc_all <- ccc_all %>%
    mutate(Model = tolower(Model),
           Group = case_when(
             Model %in% stable_models   ~ "Stable",
             Model %in% volatile_models ~ "Volatile",
             TRUE ~ "Other"
           ))
  
  
  p <- ggplot(ccc_all, aes(x = DifficultyBin, y = Accuracy,
                           group = Model, color = Group, linetype = Group)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = c("Stable" = "forestgreen", "Volatile" = "firebrick", "Other" = "grey60")) +
    scale_linetype_manual(values = c("Stable" = "solid", "Volatile" = "dashed", "Other" = "solid")) +
    theme_minimal(base_size = 13) +
    labs(
      title = paste("Classifier Characteristic Curves —", dataset, toupper(mode)),
      subtitle = "Stable vs Volatile Models",
      x = "Item difficulty bin (quintiles)",
      y = "Mean Accuracy"
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  out_file <- file.path(out_root, paste0("CCC_", dataset, "_", mode, "_", bin_var, ".png"))
  print(p)
  ggsave(out_file, p, width = 8, height = 6, dpi = 300)
  cat("✅ Saved CCC plot:", out_file, "\n")
}

# ------------------------------------------------
# 4️⃣ Example usage: stable vs volatile models
# ------------------------------------------------

datasets <- c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100")
modes    <- c("zeroshot", "trained")

imagenet_ip <- load_item_difficulties("ImageNet", "zeroshot")
global_breaks <- quantile(imagenet_ip$b_rank, probs = seq(0, 1, 0.2), na.rm = TRUE)
print(global_breaks)

for (mode in modes) {
  
  # --- Pick stable/volatile sets based on mode ---
  if (mode == "zeroshot") {
    stable_models   <- tolower(head(top_zero_acc$stable$Model, 3))
    volatile_models <- tolower(head(top_zero_acc$volatile$Model, 3))
  } else if (mode == "trained") {
    stable_models   <- tolower(head(top_mix_theta$stable$Model, 3))
    volatile_models <- tolower(head(top_mix_theta$volatile$Model, 3))
  }
  
  models_to_plot <- c(stable_models, volatile_models)
  
  # --- Run CCC plotting per dataset ---
  for (ds in datasets) {
    if (ds == "ImageNet" && mode == "trained") next
    plot_ccc(
      dataset = ds,
      mode = mode,
      models = models_to_plot,
      global_breaks = global_breaks,
      bin_var = "b_rank",
      stable_models = stable_models,
      volatile_models = volatile_models
    )
  }
}

# ------------------------------------------------
# Difficulty–Accuracy Correlation Summary + Significance Tests
# ------------------------------------------------

library(dplyr)
library(ggplot2)
library(tidyr)

b_acc_path <- "output/Hypothesis_2/analysis/concept_difficulty_vs_acc_modes/b_acc_correlations_per_model.csv"
if (!file.exists(b_acc_path)) stop("❌ Missing b–accuracy correlation file.")

b_acc <- read.csv(b_acc_path) %>%
  mutate(Model = tolower(Model))

# Container to collect all test results
pval_summary <- data.frame(
  Comparison = character(),
  Mode = character(),
  P_value = numeric(),
  stringsAsFactors = FALSE
)

summary_corr_all <- data.frame()  # store all correlations for cross-mode comparisons

for (mode in c("zeroshot", "trained")) {
  
  cat("\n=== Running difficulty–accuracy correlation summary for mode:", mode, "===\n")
  
  # --- Pick stable/volatile sets based on mode ---
  if (mode == "zeroshot") {
    stable_models   <- tolower(head(top_zero_acc$stable$Model, 10))
    volatile_models <- tolower(head(top_zero_acc$volatile$Model, 10))
  } else if (mode == "trained") {
    stable_models   <- tolower(head(top_mix_theta$stable$Model, 10))
    volatile_models <- tolower(head(top_mix_theta$volatile$Model, 10))
  }
  
  # --- Compute summary correlations ---
  summary_corr <- b_acc %>%
    filter(Mode == mode) %>%
    group_by(Model) %>%
    summarise(
      mean_rho = mean(rho, na.rm = TRUE),
      sd_rho = sd(rho, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(Group = case_when(
      Model %in% stable_models ~ "Stable",
      Model %in% volatile_models ~ "Volatile",
      TRUE ~ "Other"
    ),
    Mode = mode)
  
  summary_corr_all <- bind_rows(summary_corr_all, summary_corr)
  
  group_stats <- summary_corr %>%
    group_by(Group) %>%
    summarise(
      mean_of_means = mean(mean_rho, na.rm = TRUE),
      mean_of_sd = mean(sd_rho, na.rm = TRUE),
      .groups = "drop"
    )
  
  print(group_stats)
  
  # --- Mann–Whitney significance test (Stable vs Volatile) ---
  stable_rho <- summary_corr %>% filter(Group == "Stable") %>% pull(mean_rho)
  volatile_rho <- summary_corr %>% filter(Group == "Volatile") %>% pull(mean_rho)
  
  if (length(stable_rho) > 1 && length(volatile_rho) > 1) {
    test_result <- wilcox.test(stable_rho, volatile_rho, alternative = "two.sided", exact = FALSE)
    p_value <- test_result$p.value
    cat(sprintf("📊 %s mode: Mann–Whitney U test (Stable vs Volatile) → p = %.4f\n", mode, p_value))
    pval_summary <- rbind(pval_summary, data.frame(
      Comparison = "Stable vs Volatile",
      Mode = mode,
      P_value = p_value
    ))
  } else {
    cat(sprintf("⚠️ %s mode: insufficient models for statistical test.\n", mode))
  }
  
  # --- Optional plot ---
  ggplot(summary_corr, aes(x = mean_rho, fill = Group)) +
    geom_density(alpha = 0.4) +
    geom_vline(xintercept = 0, linetype = "dotted") +
    scale_fill_manual(values = c("Stable" = "forestgreen", "Volatile" = "firebrick", "Other" = "grey70")) +
    theme_minimal(base_size = 14) +
    labs(
      title = paste("Distribution of ρ(b, accuracy) —", toupper(mode), "mode"),
      x = "Mean Spearman correlation ρ(b, accuracy)",
      y = "Density"
    ) -> p
  
  ggsave(file.path("output/Hypothesis_2/analysis/CCC",
                   paste0("b_acc_rho_dist_", mode, ".png")), p, width = 8, height = 6, dpi = 300)
  print(p)
}


# ------------------------------------------------
# 🔄 Cross-mode comparisons (Stable & Volatile)
# ------------------------------------------------
cat("\n=== Running cross-mode significance tests ===\n")

# Extract group-wise distributions
stable_zero <- summary_corr_all %>% filter(Mode == "zeroshot", Group == "Stable") %>% pull(mean_rho)
stable_trained <- summary_corr_all %>% filter(Mode == "trained", Group == "Stable") %>% pull(mean_rho)
volatile_zero <- summary_corr_all %>% filter(Mode == "zeroshot", Group == "Volatile") %>% pull(mean_rho)
volatile_trained <- summary_corr_all %>% filter(Mode == "trained", Group == "Volatile") %>% pull(mean_rho)

# --- Stable across modes ---
if (length(stable_zero) > 1 && length(stable_trained) > 1) {
  test_stable <- wilcox.test(stable_zero, stable_trained, alternative = "two.sided", exact = FALSE)
  p_stable <- test_stable$p.value
  cat(sprintf("Stable (zeroshot vs trained) → p = %.4f\n", p_stable))
  pval_summary <- rbind(pval_summary, data.frame(
    Comparison = "Stable across modes",
    Mode = "Zero→Trained",
    P_value = p_stable
  ))
}

# --- Volatile across modes ---
if (length(volatile_zero) > 1 && length(volatile_trained) > 1) {
  test_volatile <- wilcox.test(volatile_zero, volatile_trained, alternative = "two.sided", exact = FALSE)
  p_volatile <- test_volatile$p.value
  cat(sprintf("Volatile (zeroshot vs trained) → p = %.4f\n", p_volatile))
  pval_summary <- rbind(pval_summary, data.frame(
    Comparison = "Volatile across modes",
    Mode = "Zero→Trained",
    P_value = p_volatile
  ))
}

# ------------------------------------------------
# 📊 Print summary table
# ------------------------------------------------
cat("\n=== Mann–Whitney U Test Summary ===\n")
print(pval_summary)

# ------------------------------------------------
# 📊 Print and annotate significance interpretations
# ------------------------------------------------
pval_summary <- pval_summary %>%
  mutate(Interpretation = case_when(
    P_value < 0.001 ~ "Highly significant (p < 0.001)",
    P_value < 0.01  ~ "Significant (p < 0.01)",
    P_value < 0.05  ~ "Marginally significant (p < 0.05)",
    TRUE            ~ "Not significant (n.s.)"
  ))

cat("\n=== Mann–Whitney U Test Summary with Interpretations ===\n")
print(pval_summary)

# Optional: save annotated version for thesis appendix
write.csv(pval_summary, 
          "output/Hypothesis_2/analysis/CCC/mann_whitney_summary_interpreted.csv", 
          row.names = FALSE)
cat("\n✅ Saved annotated p-value summary to analysis/CCC/mann_whitney_summary_interpreted.csv\n")

# Mann-Whitney U test:
# Tests whether two samples come from the same distribution (i.e., have the same median rank, not the same variance).
# H_0: ests whether two samples come from the same distribution (i.e., have the same median rank, not the same variance).
# H_1: The two groups come from different distributions → one group tends to have larger or smaller values than the other.
# Therefore:
# A significant p-value means the two groups are different.
# A non-significant p-value means the two groups are similar.

library(igraph)
library(ggraph)
library(tidygraph)
library(ggplot2)
library(dplyr)

# --- Define nodes and edges (same as before) ---
nodes <- tibble::tibble(
  name = c(
    "Evaluation Modes",
    "Zeroshot","Trained",
    "Binary Correctness Matrices (Model × Item)",
    "IRT 2-Parameter Logistic Model",
    "θ → Model ability (generalization)",
    "b → Item difficulty",
    "a → Item discrimination",
    "Analyses (per mode & cross-mode)",
    "• Rank correlations (θ, accuracy, b, a)",
    "• Δb / Δa vs Δaccuracy relationships",
    "• Zeroshot ↔ Trained contrasts",
    "• Architecture-family volatility (Var z)"
  ),
  group = c("A","A","A","B","C","D","D","D","E","E","E","E","E")
)

edges <- tibble::tibble(
  from = c(
    "Evaluation Modes","Evaluation Modes",
    "Zeroshot","Trained",
    "Binary Correctness Matrices (Model × Item)",
    "IRT 2-Parameter Logistic Model","IRT 2-Parameter Logistic Model","IRT 2-Parameter Logistic Model",
    "θ → Model ability (generalization)",
    "b → Item difficulty",
    "a → Item discrimination",
    "Analyses (per mode & cross-mode)",
    "Analyses (per mode & cross-mode)",
    "Analyses (per mode & cross-mode)",
    "Analyses (per mode & cross-mode)"
  ),
  to = c(
    "Zeroshot","Trained",
    "Binary Correctness Matrices (Model × Item)",
    "Binary Correctness Matrices (Model × Item)",
    "IRT 2-Parameter Logistic Model",
    "θ → Model ability (generalization)",
    "b → Item difficulty",
    "a → Item discrimination",
    "Analyses (per mode & cross-mode)",
    "Analyses (per mode & cross-mode)",
    "Analyses (per mode & cross-mode)",
    "• Rank correlations (θ, accuracy, b, a)",
    "• Δb / Δa vs Δaccuracy relationships",
    "• Zeroshot ↔ Trained contrasts",
    "• Architecture-family volatility (Var z)"
  )
)

# --- Create graph and attach fill colors ---
g <- tbl_graph(nodes = nodes, edges = edges, directed = TRUE) %>%
  activate(nodes) %>%
  mutate(fill_col = case_when(
    group == "A" ~ "#E8F0FE",  # light blue
    group == "B" ~ "#E9F7EF",  # light green
    group == "C" ~ "#FFF3E0",  # light orange
    group == "D" ~ "#F3E5F5",  # light purple
    TRUE          ~ "#F5F5F5"   # grey
  ))

# --- Plot ---
p <- ggraph(g, layout = "sugiyama") +
  geom_edge_link(arrow = arrow(length = unit(3.5, "mm")),
                 end_cap = circle(2.5, "mm"), linewidth = 0.5, color = "grey40") +
  geom_node_label(aes(label = name, fill = fill_col),
                  label.size = 0.25, size = 3.7, family = "Helvetica",
                  label.r = unit(6, "pt"), label.padding = unit(5, "pt")) +
  scale_fill_identity() +
  theme_void() +
  theme(plot.margin = margin(10, 10, 10, 10))

p

p +
  ggplot2::coord_cartesian(clip = "off") +
  theme(plot.margin = margin(40, 60, 40, 80))

out_file <- file.path(out_root, paste0("hierarchy_pipeline_final.png"))
ggsave(out_file, p, width = 10, height = 7, dpi = 300)

