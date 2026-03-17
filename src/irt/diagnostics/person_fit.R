# ============================================================
# person_fit_arch_comparison.R
# Repo-safe version for person-fit analysis (IRTNet)
# ============================================================

rm(list = ls())

# -----------------------------
# Load libraries
# -----------------------------
library(data.table)
library(dplyr)
library(ggplot2)
library(mirt)
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
theta_root <- file.path(repo_root, config$paths$theta_estimates)  # Path for theta estimates
pred_root <- file.path(repo_root, config$paths$predictions)  # Path for predictions
person_root <- file.path(repo_root, config$paths$analysis, "IRT_based", "person_fit")  # Output for person-fit analysis

# Create output directory if not exist
dir.create(person_root, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

# Rebuild X (models × items) from prediction CSV + item list
rebuild_X <- function(pred_file, sampled_items) {
  df <- read.csv(pred_file, check.names = FALSE)
  
  # Ensure the first column is "Item" and the rest are model predictions
  if (!"Item" %in% colnames(df)) {
    colnames(df)[1] <- "Item"
  }
  
  item_ids <- df$Item
  resp_mat <- df[, -(1:2), drop = FALSE]
  
  # Convert to numeric 0/1
  resp_mat <- as.data.frame(lapply(resp_mat, function(x) as.numeric(as.character(x))))
  
  # Rows = items, cols = models
  rownames(resp_mat) <- item_ids
  
  # Subset by sampled items (in same order), then transpose
  resp_sub <- resp_mat[sampled_items, , drop = FALSE]
  X <- t(as.matrix(resp_sub))  # Now rows = models, cols = items
  
  # Set rownames as model names
  rownames(X) <- colnames(resp_sub)
  X
}

# Simple 2PL fit (only for person-fit, not full pipeline)
fit_2pl <- function(X) {
  start_vals <- mirt(X, 1, itemtype = "2PL", pars = "values")
  start_vals$value[start_vals$name == "d"] <- -qlogis(
    pmin(pmax(colMeans(X), 1e-4), 1 - 1e-4)
  )
  fit <- tryCatch({
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
  fit
}

# ---------------------------------------------------------
# MAIN PERSON-FIT LOOP
# ---------------------------------------------------------
datasets <- list.dirs(theta_root, full.names = FALSE, recursive = FALSE)
datasets <- datasets[datasets != ""]

all_personfit <- list()   # 🔴 init here

for (ds in datasets) {
  cat("\n===============================\n")
  cat("Person fit – dataset:", ds, "\n")
  
  ds_dir <- file.path(theta_root, ds)
  modes  <- list.dirs(ds_dir, full.names = FALSE, recursive = FALSE)
  modes  <- modes[modes != ""]
  
  for (md in modes) {
    cat("\n--- Mode:", md, " ---\n")
    
    mode_dir <- file.path(ds_dir, md)
    best_folder <- list.dirs(mode_dir, recursive = FALSE, full.names = TRUE)
    best_folder <- best_folder[grepl("best_", best_folder)]
    
    if (length(best_folder) == 0) {
      cat("  No best_* folder found for", ds, md, "- skipping.\n")
      next
    }
    
    best_folder <- best_folder[1]
    cat("  Using folder:", best_folder, "\n")
    
    sample_file <- file.path(best_folder, "Sampled_Items.csv")
    if (!file.exists(sample_file)) {
      cat("  Sampled_Items.csv not found in", best_folder, "- skipping.\n")
      next
    }
    
    sample_df <- fread(sample_file)
    if (!"Item" %in% colnames(sample_df)) {
      stop("Sampled_Items.csv in ", best_folder, " has no 'Item' column.")
    }
    sampled_items <- sample_df$Item
    
    pred_file <- file.path(pred_root, paste0(ds, "_", md, ".csv"))
    if (!file.exists(pred_file)) {
      cat("  Prediction file", pred_file, "not found - skipping.\n")
      next
    }
    
    # Rebuild X (models × items)
    cat("  Rebuilding response matrix X (models x items)...\n")
    X <- rebuild_X(pred_file, sampled_items)
    cat("  X dims:", paste(dim(X), collapse = " x "), "\n")
    
    if (ncol(X) < 10) {
      cat("  Too few items (", ncol(X), "), skipping person fit.\n")
      next
    }
    
    # Fit 2PL
    cat("  Fitting 2PL model for person fit...\n")
    fit <- fit_2pl(X)
    if (is.null(fit)) {
      cat("  Fit failed, skipping person fit for", ds, md, "\n")
      next
    }
    
    if (!extract.mirt(fit, "converged")) {
      warning("⚠️  EM did not report convergence for person fit model: ", ds, " ", md)
    }
    
    # Person-fit stats
    cat("  Computing person-fit statistics...\n")
    pf <- tryCatch({
      personfit(fit)
    }, error = function(e) {
      warning("personfit() failed: ", conditionMessage(e))
      NULL
    })
    
    if (is.null(pf)) {
      cat("  personfit() failed for", ds, md, "\n")
      next
    }
    
    pf_df <- as.data.frame(pf)
    
    # 🔴 KEY FIX: use rownames(X) as model names, not rownames(pf_df)
    if (nrow(pf_df) != nrow(X)) {
      warning("Row mismatch between pf_df and X for ", ds, " ", md)
    }
    pf_df$Model   <- rownames(X)
    pf_df$Dataset <- ds
    pf_df$Mode    <- md
    
    pf_df <- pf_df %>%
      dplyr::relocate(Model, Dataset, Mode)
    
    # Save per-dataset/mode CSV
    out_dir_dm <- file.path(person_root, ds)
    dir.create(out_dir_dm, recursive = TRUE, showWarnings = FALSE)
    
    out_file_dm <- file.path(out_dir_dm, paste0("person_fit_", ds, "_", md, ".csv"))
    fwrite(pf_df, out_file_dm)
    cat("  Saved person-fit CSV to:", out_file_dm, "\n")
    
    # Optional Zh histogram
    if ("Zh" %in% colnames(pf_df)) {
      p_hist <- ggplot(pf_df, aes(x = Zh)) +
        geom_histogram(bins = 20, color = "black", fill = "grey80") +
        theme_minimal() +
        labs(
          title = paste0("Person-fit Zh distribution – ", ds, " (", md, ")"),
          x = "Zh", y = "Number of models"
        )
      print(p_hist)
      plot_file <- file.path(out_dir_dm, paste0("hist_Zh_", ds, "_", md, ".png"))
      ggsave(plot_file, p_hist, width = 6, height = 4, dpi = 300)
      cat("  Saved Zh histogram to:", plot_file, "\n")
    }
    
    all_personfit[[paste(ds, md, sep = "_")]] <- pf_df
  }
}

# Combine across datasets/modes
if (length(all_personfit) > 0) {
  pf_all <- dplyr::bind_rows(all_personfit)
  out_all <- file.path(person_root, "person_fit_all_datasets_modes.csv")
  fwrite(pf_all, out_all)
  cat("\n======================================\n")
  cat("Combined person-fit saved to:", out_all, "\n")
} else {
  cat("\nNo person-fit results generated.\n")
}

cat("\nPerson-fit analysis complete.\n")


##################### ARCHITECTURE ###########################

library(dplyr)
library(ggplot2)
library(data.table)
library(purrr)
library(stringr)

#--------------------------------------------------
# Broad architecture families
#--------------------------------------------------
map_family_broad <- function(model) {
  case_when(
    # 1) CNN-family (all conv-heavy)
    grepl("convnext|resnet|resnext|repvgg|botnet|dpn|hrnet",
          model, ignore.case = TRUE) ~ "CNN-family",
    grepl("densenet", model, ignore.case = TRUE) ~ "CNN-family",
    grepl("efficientnet|mixnet|efficientformer",
          model, ignore.case = TRUE) ~ "CNN-family",
    grepl("mobilenet|mobilevit|shufflenet|mnasnet|ese_vovnet|regnetz",
          model, ignore.case = TRUE) ~ "CNN-family",
    grepl("inception|googlenet|xception|darknet",
          model, ignore.case = TRUE) ~ "CNN-family",
    grepl("alexnet|vgg", model, ignore.case = TRUE) ~ "CNN-family",
    
    # 2) ViT-family (pure transformers)
    grepl("vit|deit|beit|cait|crossvit|levit",
          model, ignore.case = TRUE) ~ "ViT-family",
    
    # 3) Hybrid-ViT (CNN + transformer style)
    grepl("swin|maxvit|xcit|davit|volo|eva|coatnet|coat",
          model, ignore.case = TRUE) ~ "Hybrid-ViT",
    
    # 4) MLP-like
    grepl("mixer|resmlp", model, ignore.case = TRUE) ~ "MLP-like",
    
    # 5) Everything else
    TRUE ~ "Other"
  )
}

pf_arch <- pf_all %>%
  mutate(
    ArchFamily = map_family_broad(Model)
  )


# Archiecture wise summaries:
arch_summary <- pf_arch %>%
  group_by(Dataset, Mode, ArchFamily) %>%
  summarise(
    n_models  = n(),
    mean_Zh   = mean(Zh, na.rm = TRUE),
    sd_Zh     = sd(Zh, na.rm = TRUE),
    median_Zh = median(Zh, na.rm = TRUE),
    IQR_Zh    = IQR(Zh, na.rm = TRUE),
    .groups   = "drop"
  ) %>%
  arrange(Dataset, Mode, ArchFamily)
# Save table
arch_summary_file <- file.path(
  person_root, "person_fit_architecture_summary.csv"
)
fwrite(arch_summary, arch_summary_file)
cat("Architecture-wise Zh summary saved to:", arch_summary_file, "\n")

############################################################
## SECTION: Architecture-wise person-fit summaries & plots
## (run after pf_arch and arch_summary are created)
############################################################

library(dplyr)
library(tidyr)
library(ggplot2)
library(viridis)
library(ggridges)   # for ridgeplots

# Consistent ordering for datasets and modes
ds_levels   <- c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100")
mode_levels <- c("zeroshot", "trained")

arch_summary <- arch_summary %>%
  mutate(
    Dataset = factor(Dataset, levels = ds_levels),
    Mode    = factor(Mode,    levels = mode_levels)
  )

pf_arch <- pf_arch %>%
  mutate(
    Dataset = factor(Dataset, levels = ds_levels),
    Mode    = factor(Mode,    levels = mode_levels)
  )

out_dir_arch <- file.path(person_root, "architecture_plots")
dir.create(out_dir_arch, recursive = TRUE, showWarnings = FALSE)
############################################################
## "Architecture Stability Fingerprints"
##           – mean Zh profile across datasets
############################################################

profile_data <- arch_summary %>%
  filter(n_models >= 3) %>%  # optional: keep families with at least 3 models
  arrange(Dataset)

p_profile <- ggplot(
  profile_data,
  aes(x = Dataset, y = mean_Zh, group = ArchFamily, color = ArchFamily)
) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~ Mode, ncol = 2) +
  scale_color_viridis_d(end = 0.9, name = "Architecture\nfamily") +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    axis.title.x     = element_blank(),
    strip.text       = element_text(face = "bold"),
    legend.position  = "bottom"
  ) +
  labs(
    title = "Architecture stability fingerprints (Zh)",
    y     = "Mean Zh per architecture family"
  )

print(p_profile)
ggsave(
  filename = file.path(out_dir_arch, "arch_profile_mean_Zh.png"),
  plot     = p_profile,
  width    = 8,
  height   = 5,
  dpi      = 300
)

# ==========================================================
# PART C — Relating rank volatility to person-fit (Zh)
# ==========================================================
# Idea:
#  - Use the original 2PL ability estimates (Theta) per dataset/mode
#  - Convert Theta to ranks per (Dataset, Mode)
#  - For each Mode and each reference Dataset, define a model-wise
#    "rank volatility" = SD of rank differences between that ref
#    dataset and all other datasets in that mode
#  - Correlate this volatility with Zh on the reference dataset
# =========================================================

library(data.table)
library(dplyr)
library(tidyr)
library(ggplot2)
library(purrr)

# -------------------------------
# 1) Load person-fit results (Zh)
# -------------------------------
pf_file <- file.path(person_root, "person_fit_all_datasets_modes.csv")
if (!file.exists(pf_file)) {
  stop("person_fit_all_datasets_modes.csv not found at: ", pf_file)
}

pf_all <- fread(pf_file) %>%
  as_tibble()

# Sanity: expect columns including Model, Dataset, Mode, Zh
if (!all(c("Model", "Dataset", "Mode", "Zh") %in% colnames(pf_all))) {
  stop("pf_all does not contain expected columns: Model, Dataset, Mode, Zh.")
}

# Ensure Model is character (to match theta files)
pf_all <- pf_all %>%
  mutate(
    Model   = as.character(Model),
    Dataset = as.character(Dataset),
    Mode    = as.character(Mode)
  )

# ---------------------------------------------
# 2) Load Theta (ability) from original 2PL fits
# ---------------------------------------------
theta_list <- list()

datasets_theta <- list.dirs(theta_root, full.names = FALSE, recursive = FALSE)
datasets_theta <- datasets_theta[datasets_theta != ""]

for (ds in datasets_theta) {
  ds_dir <- file.path(theta_root, ds)
  modes <- list.dirs(ds_dir, full.names = FALSE, recursive = FALSE)
  modes <- modes[modes != ""]
  
  for (md in modes) {
    mode_dir <- file.path(ds_dir, md)
    best_folder <- list.dirs(mode_dir, recursive = FALSE, full.names = TRUE)
    best_folder <- best_folder[grepl("best_", best_folder)]
    
    if (length(best_folder) == 0) {
      message("No best_* folder for Theta in ", ds, " / ", md, " – skipping.")
      next
    }
    
    best_folder <- best_folder[1]
    theta_file  <- file.path(best_folder, "Theta_ModelAbilities_Long.csv")
    if (!file.exists(theta_file)) {
      message("Theta_ModelAbilities_Long.csv missing for ", ds, " / ", md, " – skipping.")
      next
    }
    
    th <- fread(theta_file) %>% as_tibble()
    
    # Expect columns: Model, Theta, Dataset (from your fitting script)
    if (!all(c("Model", "Theta") %in% colnames(th))) {
      stop("Theta file ", theta_file, " does not contain Model and Theta columns.")
    }
    
    th <- th %>%
      mutate(
        Model   = as.character(Model),
        Dataset = ds,         # enforce consistency
        Mode    = md
      )
    
    theta_list[[paste(ds, md, sep = "_")]] <- th
  }
}

if (length(theta_list) == 0) {
  stop("No Theta_ModelAbilities_Long.csv files found – cannot compute volatility.")
}

theta_long <- bind_rows(theta_list)

# -------------------------------------------------
# 3) Convert Theta to ranks per (Dataset, Mode)
# -------------------------------------------------
theta_ranks <- theta_long %>%
  group_by(Dataset, Mode) %>%
  mutate(
    rank_theta = rank(-Theta, ties.method = "average")  # higher Theta => smaller rank
  ) %>%
  ungroup()

# -----------------------------------------------------------
# 4) Compute dataset-relative rank volatility (Option C)
# -----------------------------------------------------------
# For each Mode and each reference Dataset, define for each Model:
#   volatility(Model | Mode, RefDataset) =
#       SD over other datasets d of [ rank(Model,d) − rank(Model,RefDataset) ]

vol_list <- list()
modes_present <- unique(theta_ranks$Mode)

for (md in modes_present) {
  sub_md <- theta_ranks %>% filter(Mode == md)
  
  # wide matrix: rows = models, columns = datasets, entries = ranks
  ranks_wide <- sub_md %>%
    select(Model, Dataset, rank_theta) %>%
    distinct() %>%
    pivot_wider(names_from = Dataset, values_from = rank_theta)
  
  # Keep the list of datasets actually present in this mode
  ds_in_mode <- setdiff(colnames(ranks_wide), "Model")
  
  for (ref_ds in ds_in_mode) {
    other_ds <- setdiff(ds_in_mode, ref_ds)
    if (length(other_ds) < 1) {
      # With only the reference dataset, volatility is undefined
      next
    }
    
    vol_vec <- numeric(nrow(ranks_wide))
    
    for (i in seq_len(nrow(ranks_wide))) {
      r_ref  <- ranks_wide[[ref_ds]][i]
      # rank differences to all other datasets
      diffs  <- as.numeric(ranks_wide[i, other_ds, drop = TRUE]) - r_ref
      diffs  <- diffs[!is.na(diffs)]
      if (length(diffs) >= 1) {
        vol_vec[i] <- stats::sd(diffs)
      } else {
        vol_vec[i] <- NA_real_
      }
    }
    
    df_ref <- tibble(
      Model      = ranks_wide$Model,
      Mode       = md,
      RefDataset = ref_ds,
      Volatility = vol_vec
    )
    
    vol_list[[paste(md, ref_ds, sep = "_")]] <- df_ref
  }
}

if (length(vol_list) == 0) {
  stop("No rank-volatility results computed – check theta_ranks content.")
}

vol_df <- bind_rows(vol_list)

# -----------------------------------------------------------
# 5) Attach Zh (on the same reference dataset) and summarise
# -----------------------------------------------------------
# For each (Model, Mode, RefDataset), grab Zh from pf_all
# on that same Dataset = RefDataset.

vol_with_zh <- vol_df %>%
  left_join(
    pf_all %>%
      select(Model, Dataset, Mode, Zh),
    by = c("Model", "Mode", "RefDataset" = "Dataset")
  )

# Optional: define a "misfit score" = -Zh (higher => worse fit)
vol_with_zh <- vol_with_zh %>%
  mutate(
    Misfit = -Zh
  )

# Summary correlations per Mode × RefDataset
cor_summary <- vol_with_zh %>%
  group_by(Mode, RefDataset) %>%
  summarise(
    n_pairs      = sum(!is.na(Volatility) & !is.na(Zh)),
    spearman_Zh  = suppressWarnings(cor(Volatility, Zh,
                                        method = "spearman",
                                        use    = "complete.obs")),
    pearson_Zh   = suppressWarnings(cor(Volatility, Zh,
                                        method = "pearson",
                                        use    = "complete.obs")),
    spearman_misfit = suppressWarnings(cor(Volatility, Misfit,
                                           method = "spearman",
                                           use    = "complete.obs")),
    pearson_misfit  = suppressWarnings(cor(Volatility, Misfit,
                                           method = "pearson",
                                           use    = "complete.obs")),
    .groups = "drop"
  ) %>%
  arrange(Mode, RefDataset)

print(cor_summary)

# Save summary table
cor_outfile <- file.path(person_root, "Zh_vs_rank_volatility_summary.csv")
fwrite(cor_summary, cor_outfile)
cat("\nSaved Zh–volatility correlation summary to:\n  ", cor_outfile, "\n")

# -----------------------------------------------------------
# 6) Scatter plots: Zh vs rank volatility
# -----------------------------------------------------------
plot_dir <- file.path(person_root, "Zh_vs_volatility_plots")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

for (md in unique(vol_with_zh$Mode)) {
  for (ref_ds in unique(vol_with_zh$RefDataset[vol_with_zh$Mode == md])) {
    
    sub <- vol_with_zh %>%
      filter(Mode == md, RefDataset == ref_ds,
             !is.na(Volatility), !is.na(Zh))
    
    if (nrow(sub) < 3) next  # too few points for a useful scatter
    
    # For convenience, fetch the Spearman rho from the summary table
    rho <- cor_summary %>%
      filter(Mode == md, RefDataset == ref_ds) %>%
      pull(spearman_Zh)
    
    p <- ggplot(sub, aes(x = Zh, y = Volatility)) +
      geom_point(size = 2, alpha = 0.85) +
      geom_smooth(method = "lm", se = FALSE,
                  linewidth = 0.7, linetype = "dashed") +
      theme_minimal() +
      labs(
        title = paste0("Zh vs rank volatility – ", md,
                       " (ref = ", ref_ds, ")"),
        subtitle = paste0("Spearman ρ ≈ ",
                          sprintf("%.3f", rho)),
        x = "Person-fit Zh on reference dataset",
        y = "Rank volatility (SD of rank differences)"
      )
    
    print(p)
    
    out_plot <- file.path(
      plot_dir,
      paste0("Zh_vs_volatility_", md, "_ref_", ref_ds, ".png")
    )
    ggsave(out_plot, p, width = 6, height = 4, dpi = 300)
    cat("Saved plot:", out_plot, "\n")
  }
}

cat("\nRank volatility vs Zh analysis complete.\n")


############################################################
# PART D — Match the THESIS definitions:
#   misfit(model, regime) = mean_d |Zh|
#   rank instability(model, regime) = SD(rank_d) and range(rank_d)
# Regimes:
#   - Zero-shot: ImageNet, ImageNet-C, Sketch, CIFAR100 all in zeroshot
#   - Trained: ImageNet kept as zeroshot reference + (ImageNet-C, Sketch, CIFAR100) trained
############################################################

# -----------------------------
# 1) Build a "Regime" view for person-fit (pf_all)
# -----------------------------
# pf_all has: Model, Dataset, Mode, Zh
# We need a regime-level average |Zh| across datasets.

pf_all_regime <- pf_all %>%
  mutate(
    # define "regime" exactly as in the thesis design
    Regime = case_when(
      Mode == "zeroshot" ~ "Zero-shot",
      Mode == "trained"  ~ "Trained",
      TRUE               ~ as.character(Mode)
    )
  )

# Add ImageNet zeroshot into the Trained regime as the reference domain
pf_imagenet_zs_as_tr <- pf_all_regime %>%
  filter(Dataset == "ImageNet", Regime == "Zero-shot") %>%
  mutate(Regime = "Trained")

pf_all_regime2 <- bind_rows(pf_all_regime, pf_imagenet_zs_as_tr)

# Keep only the dataset sets that define each regime
pf_all_regime2 <- pf_all_regime2 %>%
  filter(
    (Regime == "Zero-shot" & Dataset %in% c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100")) |
      (Regime == "Trained" & Dataset %in% c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100"))
  )

# Compute mean absolute |Zh| across datasets within each regime
misfit_by_model <- pf_all_regime2 %>%
  group_by(Regime, Model) %>%
  summarise(
    mean_abs_Zh = mean(abs(Zh), na.rm = TRUE),
    n_datasets  = sum(!is.na(Zh)),
    .groups     = "drop"
  )

# -----------------------------
# 2) Build a "Regime" view for theta ranks
# -----------------------------
# theta_ranks has: Model, Dataset, Mode, Theta, rank_theta
theta_ranks_regime <- theta_ranks %>%
  mutate(
    Regime = case_when(
      Mode == "zeroshot" ~ "Zero-shot",
      Mode == "trained"  ~ "Trained",
      TRUE               ~ as.character(Mode)
    )
  )

# Add ImageNet zeroshot ranks into the Trained regime as the reference
theta_imagenet_zs_as_tr <- theta_ranks_regime %>%
  filter(Dataset == "ImageNet", Regime == "Zero-shot") %>%
  mutate(Regime = "Trained")

theta_ranks_regime2 <- bind_rows(theta_ranks_regime, theta_imagenet_zs_as_tr)

# Keep only the datasets that define each regime
theta_ranks_regime2 <- theta_ranks_regime2 %>%
  filter(
    (Regime == "Zero-shot" & Dataset %in% c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100")) |
      (Regime == "Trained" & Dataset %in% c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100"))
  )

# -----------------------------
# 3) Compute rank SD and rank range across datasets (per model, per regime)
# -----------------------------
rank_instability_by_model <- theta_ranks_regime2 %>%
  group_by(Regime, Model) %>%
  summarise(
    rank_sd    = sd(rank_theta, na.rm = TRUE),
    rank_range = (max(rank_theta, na.rm = TRUE) - min(rank_theta, na.rm = TRUE)),
    n_datasets = sum(!is.na(rank_theta)),
    .groups    = "drop"
  )

# -----------------------------
# 4) Join misfit + rank instability, and compute correlations per regime
# -----------------------------
misfit_rank_df <- rank_instability_by_model %>%
  inner_join(misfit_by_model, by = c("Regime", "Model"), suffix = c("_rank", "_misfit"))

# Correlations that match the thesis table
cor_table <- misfit_rank_df %>%
  group_by(Regime) %>%
  summarise(
    n_models = sum(!is.na(mean_abs_Zh) & !is.na(rank_sd) & !is.na(rank_range)),
    rho_absZh_rankSD = suppressWarnings(cor(mean_abs_Zh, rank_sd,
                                            method = "spearman", use = "complete.obs")),
    rho_absZh_rankRange = suppressWarnings(cor(mean_abs_Zh, rank_range,
                                               method = "spearman", use = "complete.obs")),
    .groups = "drop"
  )

print(cor_table)

fwrite(misfit_rank_df, file.path(person_root, "model_misfit_and_rank_instability_by_regime.csv"))
fwrite(cor_table,      file.path(person_root, "misfit_vs_rank_instability_correlations_by_regime.csv"))

cat("\nSaved thesis-matching misfit + instability tables to:\n  ", out_dir_thesis, "\n")
