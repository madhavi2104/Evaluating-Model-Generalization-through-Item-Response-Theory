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



                                   
# [The rest of the code follows for rank volatility analysis...]
