# ==========================================================
# IRT ANALYSIS: DIF (within dataset), Linking (across datasets),
#               and Test Information Functions (TIF)
# ==========================================================
# - Part A: DIF (zeroshot vs trained) within each dataset
# - Part B: IRT Linking / Test Equating to ImageNet (zeroshot)
# - Part C: Test Information Function comparison across datasets
# ==========================================================
rm(list = ls())

library(dplyr)
library(ggplot2)
library(readr)
library(tibble)
library(stringr)
library(tidyr)
library(viridis)

# -----------------------------
# Load configuration
# -----------------------------
config <- yaml.load_file("config/config.yml")

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."

# --- Directories ---
pred_root   <- file.path(repo_root, config$paths$predictions)  # Use config path for predictions
irt_root    <- file.path(repo_root, config$paths$theta_estimates)  # Use config path for theta estimates
dif_root    <- file.path(repo_root, config$paths$analysis, "irt", "dif_within_dataset")  # Use config path for DIF
link_root   <- file.path(repo_root, config$paths$analysis, "irt", "irt_linking")  # Use config path for linking
tif_root    <- file.path(repo_root, config$paths$analysis, "irt", "test_information")  # Use config path for TIF

dir.create(dif_root,  recursive = TRUE, showWarnings = FALSE)
dir.create(link_root, recursive = TRUE, showWarnings = FALSE)
dir.create(tif_root,  recursive = TRUE, showWarnings = FALSE)

datasets_dif   <- c("Sketch", "ImageNet-C", "CIFAR100")
datasets_all   <- c("ImageNet", "Sketch", "ImageNet-C", "CIFAR100")
modes_to_check <- c("zeroshot", "trained")

# ==========================================================
# 1️⃣ Mapping tables (ImageNet ↔ CIFAR superclasses)
#    (used in DIF part for superclass summaries)
# ==========================================================

map_df <- read.csv(config$datasets$imagenet_to_cifar_superclasses.csv, stringsAsFactors = FALSE) %>%
  rename(
    imagenet_id      = ImageNet_ID,
    imagenet_class   = ImageNet_Class,
    cifar_superclass = CIFAR_Superclass
  ) %>%
  mutate(across(everything(), tolower))

cifar_map <- tribble(
  ~Fine, ~Superclass,
  # aquatic mammals
  "beaver", "aquatic mammals",
  "dolphin", "aquatic mammals",
  "otter", "aquatic mammals",
  "seal", "aquatic mammals",
  "whale", "aquatic mammals",
  # fish
  "aquarium_fish", "fish",
  "flatfish", "fish",
  "ray", "fish",
  "shark", "fish",
  "trout", "fish",
  # flowers
  "orchid", "flowers",
  "poppy", "flowers",
  "rose", "flowers",
  "sunflower", "flowers",
  "tulip", "flowers",
  # food containers
  "bottle", "food containers",
  "bowl", "food containers",
  "can", "food containers",
  "cup", "food containers",
  "plate", "food containers",
  # fruit and vegetables
  "apple", "fruit and vegetables",
  "mushroom", "fruit and vegetables",
  "orange", "fruit and vegetables",
  "pear", "fruit and vegetables",
  "sweet_pepper", "fruit and vegetables",
  # household electrical devices
  "clock", "household electrical devices",
  "computer_keyboard", "household electrical devices",
  "lamp", "household electrical devices",
  "telephone", "household electrical devices",
  "television", "household electrical devices",
  # household furniture
  "bed", "household furniture",
  "chair", "household furniture",
  "couch", "household furniture",
  "table", "household furniture",
  "wardrobe", "household furniture",
  # insects
  "bee", "insects",
  "beetle", "insects",
  "butterfly", "insects",
  "caterpillar", "insects",
  "cockroach", "insects",
  # large carnivores
  "bear", "large carnivores",
  "leopard", "large carnivores",
  "lion", "large carnivores",
  "tiger", "large carnivores",
  "wolf", "large carnivores",
  # large man-made outdoor things
  "bridge", "large man-made outdoor things",
  "castle", "large man-made outdoor things",
  "house", "large man-made outdoor things",
  "road", "large man-made outdoor things",
  "skyscraper", "large man-made outdoor things",
  # large natural outdoor scenes
  "cloud", "large natural outdoor scenes",
  "forest", "large natural outdoor scenes",
  "mountain", "large natural outdoor scenes",
  "plain", "large natural outdoor scenes",
  "sea", "large natural outdoor scenes",
  # large omnivores and herbivores
  "camel", "large omnivores and herbivores",
  "cattle", "large omnivores and herbivores",
  "chimpanzee", "large omnivores and herbivores",
  "elephant", "large omnivores and herbivores",
  "kangaroo", "large omnivores and herbivores",
  # medium-sized mammals
  "fox", "medium-sized mammals",
  "porcupine", "medium-sized mammals",
  "possum", "medium-sized mammals",
  "raccoon", "medium-sized mammals",
  "skunk", "medium-sized mammals",
  # non-insect invertebrates
  "crab", "non-insect invertebrates",
  "lobster", "non-insect invertebrates",
  "snail", "non-insect invertebrates",
  "spider", "non-insect invertebrates",
  "worm", "non-insect invertebrates",
  # people
  "baby", "people",
  "boy", "people",
  "girl", "people",
  "man", "people",
  "woman", "people",
  # reptiles
  "crocodile", "reptiles",
  "dinosaur", "reptiles",
  "lizard", "reptiles",
  "snake", "reptiles",
  "turtle", "reptiles",
  # small mammals
  "hamster", "small mammals",
  "mouse", "small mammals",
  "rabbit", "small mammals",
  "shrew", "small mammals",
  "squirrel", "small mammals",
  # trees
  "maple_tree", "trees",
  "oak_tree", "trees",
  "palm_tree", "trees",
  "pine_tree", "trees",
  "willow_tree", "trees",
  # vehicles 1
  "bicycle", "vehicles 1",
  "bus", "vehicles 1",
  "motorcycle", "vehicles 1",
  "pickup_truck", "vehicles 1",
  "train", "vehicles 1",
  # vehicles 2
  "lawn_mower", "vehicles 2",
  "rocket", "vehicles 2",
  "streetcar", "vehicles 2",
  "tank", "vehicles 2",
  "tractor", "vehicles 2"
) %>%
  mutate(across(everything(), tolower))

# ==========================================================
# 2️⃣ Helper functions
# ==========================================================

extract_class <- function(item) {
  gsub("/.*", "", item)
}

# --- Load item parameters (with a1/d → a/b and orientation check) ---
load_item_params <- function(dataset, mode,
                             pred_root,
                             n_boot = 1000) {
  
  base_dir <- file.path(irt_root, dataset, mode)
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

  
  # --- Rename columns as per your RDS structure ---
  if ("a1" %in% names(ip)) {
    names(ip)[names(ip) == "a1"] <- "a"
  }
  if (!"a" %in% names(ip)) {
    ip$a <- NA_real_
  }

  if ("d" %in% names(ip)) {
    names(ip)[names(ip) == "d"] <- "b"
  }
  if (!"b" %in% names(ip)) {
    stop("❌ No difficulty (b) column found in ", dataset, " (", mode, ")")
  }
  
  
  # --- Orientation check for b vs accuracy ---
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
    
    rho_obs <- suppressWarnings(cor(merged$b, merged$p_value,
                                    method = "spearman", use = "complete.obs"))
    
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
    
    if (!is.na(rho_obs) && !any(is.na(ci)) && ci[1] > 0) {
      ip$b <- -ip$b
      message(sprintf("🔄 Flipped difficulty orientation for %s (%s) [ρ=%.3f, 95%% CI (%.3f – %.3f)]",
                      dataset, mode, rho_obs, ci[1], ci[2]))
    } else {
      message(sprintf("✅ Difficulty orientation OK for %s (%s) [ρ=%.3f, 95%% CI (%.3f – %.3f)]",
                      dataset, mode, rho_obs, ci[1], ci[2]))
    }
  } else {
    message("⚠️ Skipping orientation check — no predictions found for ", dataset, " (", mode, ")")
  }
  
  ip$Dataset <- dataset
  ip$Mode    <- mode
  return(ip)
}

# --- Per-item significance for S-DIF via bootstrap CI of Δ parameter ---
# Returns logical sig flag and mean magnitude |Δ|
bootstrap_item_sig <- function(delta_vec, B = 2000, alpha = 0.05) {
  delta_vec <- delta_vec[!is.na(delta_vec)]
  n <- length(delta_vec)
  if (n < 5) {
    return(list(sig = NA, mag = NA))
  }
  
  set.seed(123)
  boot_means <- replicate(B, mean(sample(delta_vec, n, replace = TRUE)))
  ci <- quantile(boot_means, probs = c(alpha/2, 1 - alpha/2), na.rm = TRUE)
  
  # "significant" if CI excludes 0
  sig <- (ci[1] > 0) || (ci[2] < 0)
  
  list(sig = sig, mag = abs(mean(delta_vec, na.rm = TRUE)))
}


# ==========================================================
# PART A — DIF (zeroshot vs trained) WITHIN DATASET
# ==========================================================

compute_dif_for_dataset <- function(dataset) {
  
  message("=== DIF analysis (zeroshot vs trained) for: ", dataset, " ===")
  
  ip_zs <- load_item_params(dataset, "zeroshot")
  ip_tr <- load_item_params(dataset, "trained")
  
  if (is.null(ip_zs) || is.null(ip_tr)) {
    message("⚠️ Missing parameters for ", dataset, " — skipping DIF.")
    return(NULL)
  }
  
  # --- Overlapping items only ---
  common_items <- intersect(ip_zs$Item, ip_tr$Item)
  n_common     <- length(common_items)
  message(sprintf("Found %d common items between zeroshot and trained for %s.",
                  n_common, dataset))
  
  if (n_common == 0) {
    message("❌ No overlapping items; cannot compute DIF for ", dataset)
    return(NULL)
  }
  
  prop_zs <- n_common / nrow(ip_zs)
  prop_tr <- n_common / nrow(ip_tr)
  message(sprintf("Overlap: %.1f%% of zeroshot items, %.1f%% of trained items.",
                  100 * prop_zs, 100 * prop_tr))
  
  ip_zs <- ip_zs %>% filter(Item %in% common_items)
  ip_tr <- ip_tr %>% filter(Item %in% common_items)
  
  # Ensure we always have an 'a' column (already done in loader, but just in case)
  if (!"a" %in% names(ip_zs)) ip_zs$a <- NA_real_
  if (!"a" %in% names(ip_tr)) ip_tr$a <- NA_real_
  
  dif_items <- ip_zs %>%
    select(Item, b_zs = b, a_zs = a) %>%
    inner_join(
      ip_tr %>% select(Item, b_tr = b, a_tr = a),
      by = "Item"
    ) %>%
    mutate(
      delta_b = b_tr - b_zs,
      delta_a = a_tr - a_zs
    )
  # ---------------------------------------------------------
  # Significance of S-DIF at item-level (bootstrap CI excludes 0)
  # Interpretation: item shows systematic shift (not just noise)
  # ---------------------------------------------------------
  # Here "groups" are the two regimes; we use Δ directly, so significance is about Δ != 0.
  # If you want stricter testing, we can do permutation tests later.
  B_boot <- 2000
  
  sig_b <- bootstrap_item_sig(dif_items$delta_b, B = B_boot)
  sig_a <- bootstrap_item_sig(dif_items$delta_a, B = B_boot)
  
  frac_sig_b <- mean(dif_items$delta_b != 0, na.rm = TRUE)  # placeholder if you want per-item tests
  # Better: "fraction significant" should be per-item, not global. We'll do per-item significance by thresholding:
  # ---------------------------------------------------------
  # Structural DIF flag (distribution-based outlier rule)
  # "Significant S-DIF" = item has an unusually large signed shift
  # relative to the overall Δ distribution within the dataset.
  # ---------------------------------------------------------
  dif_items <- dif_items %>%
    mutate(
      z_db = (delta_b - mean(delta_b, na.rm = TRUE)) / sd(delta_b, na.rm = TRUE),
      z_da = (delta_a - mean(delta_a, na.rm = TRUE)) / sd(delta_a, na.rm = TRUE),
      sig_sdif_b = abs(z_db) >= 2,
      sig_sdif_a = abs(z_da) >= 2
    )
  
  S_DIF_b <- mean(dif_items$delta_b, na.rm = TRUE)
  U_DIF_b <- mean(abs(dif_items$delta_b), na.rm = TRUE)
  S_DIF_a <- mean(dif_items$delta_a, na.rm = TRUE)
  U_DIF_a <- mean(abs(dif_items$delta_a), na.rm = TRUE)
  
  message(sprintf("  Difficulty DIF (b): S-DIF = %.3f, U-DIF = %.3f",
                  S_DIF_b, U_DIF_b))
  message(sprintf("  Discrimin. DIF (a): S-DIF = %.3f, U-DIF = %.3f",
                  S_DIF_a, U_DIF_a))
  
  write.csv(
    dif_items,
    file = file.path(dif_root, paste0("dif_items_", dataset, ".csv")),
    row.names = FALSE
  )
  
  # Histograms for Δb and Δa
  p_db <- ggplot(dif_items, aes(x = delta_b)) +
    geom_histogram(bins = 40) +
    theme_minimal() +
    labs(
      title = paste0(dataset, " – Item-level Δb (trained – zeroshot)"),
      subtitle = sprintf("S-DIF = %.3f, U-DIF = %.3f", S_DIF_b, U_DIF_b),
      x = expression(Delta * "b"),
      y = "Count"
    )
  
  ggsave(
    filename = file.path(dif_root, paste0("hist_delta_b_", dataset, ".png")),
    plot     = p_db,
    width    = 7, height = 5, dpi = 300
  )
  
  if (!all(is.na(dif_items$delta_a))) {
    p_da <- ggplot(dif_items, aes(x = delta_a)) +
      geom_histogram(bins = 40) +
      theme_minimal() +
      labs(
        title = paste0(dataset, " – Item-level Δa (trained – zeroshot)"),
        subtitle = sprintf("S-DIF = %.3f, U-DIF = %.3f", S_DIF_a, U_DIF_a),
        x = expression(Delta * "a"),
        y = "Count"
      )
    
    ggsave(
      filename = file.path(dif_root, paste0("hist_delta_a_", dataset, ".png")),
      plot     = p_da,
      width    = 7, height = 5, dpi = 300
    )
  } else {
    message("  ⚠️ Discrimination 'a' not available or all NA — skipping Δa histogram.")
  }
  
  # Superclass-level DIF (optional but nice for interpretation)
  dif_items <- dif_items %>%
    mutate(Class = extract_class(Item))
  
  dif_items <- dif_items %>%
    left_join(map_df, by = c("Class" = "imagenet_id")) %>%
    mutate(
      WordNetClass = ifelse(!is.na(imagenet_class), imagenet_class, Class),
      Superclass   = cifar_superclass
    )
  
  if (dataset == "CIFAR100") {
    dif_items <- dif_items %>%
      left_join(cifar_map, by = c("Class" = "Fine")) %>%
      mutate(Superclass = ifelse(!is.na(Superclass.y), Superclass.y, Superclass.x)) %>%
      select(-Superclass.x, -Superclass.y)
  }
  
  dif_items <- dif_items %>%
    mutate(across(c(WordNetClass, Superclass), ~ tolower(trimws(.))))
  
  dif_super <- dif_items %>%
    group_by(Superclass) %>%
    summarise(
      mean_delta_b = mean(delta_b, na.rm = TRUE),
      mean_delta_a = mean(delta_a, na.rm = TRUE),
      n_items      = n(),
      .groups      = "drop"
    ) %>%
    arrange(desc(abs(mean_delta_b)))
  
  write.csv(
    dif_super,
    file = file.path(dif_root, paste0("dif_superclasses_", dataset, ".csv")),
    row.names = FALSE
  )
  
  # Barplot of top superclasses by |mean Δb|
  top_k <- 15
  dif_super_top <- dif_super %>%
    slice_head(n = top_k) %>%
    mutate(Superclass = factor(Superclass, levels = rev(Superclass)))
  
  p_super <- ggplot(dif_super_top, aes(x = Superclass, y = mean_delta_b)) +
    geom_col() +
    coord_flip() +
    theme_minimal() +
    labs(
      title = paste0(dataset, " – Superclass-level mean Δb"),
      subtitle = "Top concepts by |mean Δb| (trained – zeroshot)",
      x = "Superclass",
      y = expression("Mean " * Delta * "b")
    )
  
  ggsave(
    filename = file.path(dif_root, paste0("bar_mean_delta_b_super_", dataset, ".png")),
    plot     = p_super,
    width    = 7, height = 6, dpi = 300
  )
  # ---------------------------------------------------------
  # Summary row for thesis table
  # ---------------------------------------------------------
  frac_sig_sdif <- mean(dif_items$sig_sdif_b, na.rm = TRUE)
  mean_mag_sdif <- mean(abs(dif_items$delta_b[dif_items$sig_sdif_b]), na.rm = TRUE)
  
  summary_row <- tibble(
    Dataset = dataset,
    `Evaluation regime` = "Trained vs Zero-shot",
    `Mean |Δb|` = mean(abs(dif_items$delta_b), na.rm = TRUE),
    `Mean |Δa|` = mean(abs(dif_items$delta_a), na.rm = TRUE),
    `Fraction significant S-DIF` = frac_sig_sdif,
    `Mean magnitude of S-DIF` = mean_mag_sdif,
    n_common = n_common
  )
  
  invisible(
    list(
      dif_items  = dif_items,
      dif_super  = dif_super,
      summary_row = summary_row,
      S_DIF_b    = S_DIF_b,
      U_DIF_b    = U_DIF_b,
      S_DIF_a    = S_DIF_a,
      U_DIF_a    = U_DIF_a
    )
  )
  
}

dif_results <- lapply(datasets_dif, compute_dif_for_dataset)
names(dif_results) <- datasets_dif

# ==========================================================
# 5️⃣ Build thesis DIF summary table
# ==========================================================
dif_table <- bind_rows(lapply(dif_results, function(x) {
  if (is.null(x)) return(NULL)
  x$summary_row
}))

# Order datasets the way you want in the thesis
dif_table$Dataset <- factor(
  dif_table$Dataset,
  levels = c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100"),
  labels = c("ImageNet", "ImageNet-C", "ImageNet-Sketch", "CIFAR-100")
)

dif_table <- dif_table %>%
  arrange(Dataset)

print(dif_table)

write_csv(dif_table, file.path(dif_root, "DIF_summary_table.csv"))

# ==========================================================
# PART B — IRT LINKING / TEST EQUATING (to ImageNet zeroshot)
# ==========================================================

# 2PL probability (textbook): P = logistic(a * (theta - b))
icc_2pl <- function(theta, a, b) {
  1 / (1 + exp(-(a * (theta - b))))
  # equivalent: 1 / (1 + exp(-(a * theta - a * b)))
}

# Test characteristic curve (mean P over items)
tcc_from_items <- function(theta_grid, a, b) {
  keep <- !is.na(a) & !is.na(b)
  a <- a[keep]; b <- b[keep]
  
  sapply(theta_grid, function(th) {
    p <- icc_2pl(th, a, b)
    mean(p, na.rm = TRUE)
  })
}

# Test information function: sum_i a_i^2 * p_i * (1 - p_i)
tif_from_items <- function(theta_grid, a, b) {
  keep <- !is.na(a) & !is.na(b)
  a <- a[keep]; b <- b[keep]
  
  sapply(theta_grid, function(th) {
    p <- icc_2pl(th, a, b)
    sum((a^2) * p * (1 - p), na.rm = TRUE)
  })
}


theta_grid <- seq(-4, 4, length.out = 201)

# --- Load reference (ImageNet, zeroshot) ---
ip_ref <- load_item_params("ImageNet", "zeroshot")
if (is.null(ip_ref)) stop("❌ Could not load reference ImageNet (zeroshot) item parameters.")

mean_b_ref <- mean(ip_ref$b, na.rm = TRUE)
sd_b_ref   <- sd(ip_ref$b,   na.rm = TRUE)

tcc_ref <- tcc_from_items(theta_grid, ip_ref$a, ip_ref$b)
tif_ref <- tif_from_items(theta_grid, ip_ref$a, ip_ref$b)

# --- Linking for all datasets and modes (including zeroshot & trained) ---
link_results <- list()

for (ds in datasets_all) {
  for (md in modes_to_check) {
    
    message("=== Linking / TIF for: ", ds, " (", md, ") ===")
    
    ip_tgt <- load_item_params(ds, md)
    if (is.null(ip_tgt)) {
      message("  ⚠️ Missing item params; skipping.")
      next
    }
    # --- NEW: Discrimination distribution diagnostics -----------------
    a_df <- data.frame(
      Dataset = ds,
      Mode    = md,
      a       = ip_tgt$a
    )
    
    # Save numeric summaries (for thesis tables / supplement)
    a_summary <- a_df %>%
      summarise(
        Dataset = first(Dataset),
        Mode    = first(Mode),
        n_items = sum(!is.na(a)),
        mean_a  = mean(a, na.rm = TRUE),
        sd_a    = sd(a, na.rm = TRUE),
        p90     = quantile(a, 0.90, na.rm = TRUE),
        p99     = quantile(a, 0.99, na.rm = TRUE),
        max_a   = max(a, na.rm = TRUE)
      )
    
    # Store for later
    if (!exists("a_summaries")) a_summaries <- list()
    a_summaries[[paste(ds, md, sep = "_")]] <- a_summary
    
    # Plot histogram / density of a
    p_a <- ggplot(a_df, aes(x = a)) +
      geom_histogram(bins = 60) +
      theme_minimal() +
      labs(
        title = paste0(ds, " – discrimination a (", md, ")"),
        x = "a", y = "Item count"
      )
    print(p_a)
    ggsave(
      filename = file.path(
        tif_root,
        paste0("hist_a_", ds, "_", md, ".png")
      ),
      plot = p_a,
      width = 7, height = 5, dpi = 300
    )
    
    
    mean_b_tgt <- mean(ip_tgt$b, na.rm = TRUE)
    sd_b_tgt   <- sd(ip_tgt$b,   na.rm = TRUE)
    
    # Mean/SD linking constants (map target b onto ImageNet scale)
    A <- sd_b_ref / sd_b_tgt
    B <- mean_b_ref - A * mean_b_tgt
    
    b_linked <- A * ip_tgt$b + B
    
    # TCC for original and linked versions
    tcc_tgt_orig   <- tcc_from_items(theta_grid, ip_tgt$a, ip_tgt$b)
    tcc_tgt_linked <- tcc_from_items(theta_grid, ip_tgt$a, b_linked)
    
    # RMSD vs reference TCC
    rmsd_orig   <- sqrt(mean((tcc_tgt_orig   - tcc_ref)^2, na.rm = TRUE))
    rmsd_linked <- sqrt(mean((tcc_tgt_linked - tcc_ref)^2, na.rm = TRUE))
    
    # --- TIF for original and linked versions ---
    tif_tgt_orig   <- tif_from_items(theta_grid, ip_tgt$a, ip_tgt$b)
    tif_tgt_linked <- tif_from_items(theta_grid, ip_tgt$a, b_linked)
    
    # RMSD vs reference TIF
    tif_rmsd_orig   <- sqrt(mean((tif_tgt_orig   - tif_ref)^2, na.rm = TRUE))
    tif_rmsd_linked <- sqrt(mean((tif_tgt_linked - tif_ref)^2, na.rm = TRUE))
    
    
    link_results[[paste(ds, md, sep = "_")]] <- data.frame(
      Dataset          = ds,
      Mode             = md,
      A                = A,
      B                = B,
      RMSD_orig        = rmsd_orig,
      RMSD_linked      = rmsd_linked,
      TIF_RMSD_orig    = tif_rmsd_orig,
      TIF_RMSD_linked  = tif_rmsd_linked
    )
    
    
    # Save TCC plot (linked vs reference)
    tcc_df <- data.frame(
      theta         = theta_grid,
      TCC_ref       = tcc_ref,
      TCC_tgt_orig  = tcc_tgt_orig,
      TCC_tgt_linked = tcc_tgt_linked
    ) %>%
      tidyr::pivot_longer(
        cols      = c("TCC_ref", "TCC_tgt_orig", "TCC_tgt_linked"),
        names_to  = "Curve",
        values_to = "Value"
      )
    
    p_tcc <- ggplot(tcc_df, aes(x = theta, y = Value, color = Curve)) +
      geom_line(linewidth = 0.8) +
      theme_minimal() +
      scale_color_viridis_d(end = 0.8) +
      labs(
        title = paste0("TCC Linking: ImageNet (zeroshot) vs ", ds, " (", md, ")"),
        subtitle = sprintf("RMSD_orig = %.3f, RMSD_linked = %.3f", rmsd_orig, rmsd_linked),
        x = expression(theta),
        y = "Mean P(θ)"
      )
    print(p_tcc)
    ggsave(
      filename = file.path(
        link_root,
        paste0("tcc_link_", ds, "_", md, ".png")
      ),
      plot = p_tcc,
      width = 7, height = 5, dpi = 300
    )
    ggsave(
      filename = file.path(
        link_root,
        paste0("tcc_link_", ds, "_", md, ".pdf")
      ),
      plot = p_tcc,
      width = 7, height = 5
    )
    # --- Save TIF plot (linked vs reference) ---
    tif_df <- data.frame(
      theta          = theta_grid,
      TIF_ref        = tif_ref,
      TIF_tgt_orig   = tif_tgt_orig,
      TIF_tgt_linked = tif_tgt_linked
    ) %>%
      pivot_longer(
        cols = c("TIF_ref", "TIF_tgt_orig", "TIF_tgt_linked"),
        names_to = "Curve",
        values_to = "Value"
      )
    
    p_tif <- ggplot(tif_df, aes(x = theta, y = Value, color = Curve)) +
      geom_line(linewidth = 0.8) +
      theme_minimal() +
      scale_color_viridis_d(end = 0.8) +
      labs(
        title = paste0("TIF: ImageNet (zeroshot) vs ", ds, " (", md, ")"),
        subtitle = sprintf(
          "TIF RMSD orig = %.3f, linked = %.3f",
          tif_rmsd_orig, tif_rmsd_linked
        ),
        x = expression(theta),
        y = "Test information I(θ)"
      )
    
    ggsave(
      filename = file.path(
        tif_root,
        paste0("tif_link_", ds, "_", md, ".png")
      ),
      plot = p_tif,
      width = 7, height = 5, dpi = 300
    )
    ggsave(
      filename = file.path(
        tif_root,
        paste0("tif_link_", ds, "_", md, ".pdf")
      ),
      plot = p_tif,
      width = 7, height = 5
    )
    
  }
}

# --- Save linking summary and TIF curves ---

if (length(link_results) > 0) {
  link_summary <- do.call(rbind, link_results)
  write.csv(
    link_summary,
    file = file.path(link_root, "irt_linking_summary_ImageNet_ref.csv"),
    row.names = FALSE
  )
}


if (exists("a_summaries")) {
  a_summary_all <- bind_rows(a_summaries)
  write.csv(
    a_summary_all,
    file = file.path(tif_root, "discrimination_summary_all.csv"),
    row.names = FALSE
  )
}

# ==========================================================
# PART C — TIF: zeroshot vs trained *within* each dataset
# Thesis-ready version (UPDATED):
#   - computes + saves all curves (CSV)
#   - writes a compact diagnostics table (RMSD + peak ratio)
#   - produces ONE academic 1×3 grid figure with shared axes
#     order: ImageNet-C, ImageNet-Sketch, CIFAR-100
#   - facet strips show: (a) DatasetName, (b) DatasetName, (c) DatasetName
#   - saves PDF (vector) + PNG (preview)
# ==========================================================

library(dplyr)
library(ggplot2)
library(tidyr)

# ----------------------------------------------------------
# 1) Compute TIF curves for all datasets + modes
# ----------------------------------------------------------
tif_curves <- list()

for (ds in datasets_all) {
  for (md in modes_to_check) {
    
    ip <- load_item_params(ds, md)
    if (is.null(ip)) {
      message("⚠️ No item params for ", ds, " (", md, "); skipping TIF.")
      next
    }
    
    tif_vals <- tif_from_items(theta_grid, ip$a, ip$b)
    
    tif_curves[[paste(ds, md, sep = "_")]] <- data.frame(
      Dataset = ds,
      Mode    = md,
      theta   = theta_grid,
      TIF     = tif_vals
    )
  }
}

# ----------------------------------------------------------
# 2) Save all TIF curves to CSV (useful for appendix/reuse)
# ----------------------------------------------------------
if (length(tif_curves) > 0) {
  tif_all <- bind_rows(tif_curves)
  write.csv(
    tif_all,
    file = file.path(tif_root, "tif_curves_all_datasets_within_dataset_scale.csv"),
    row.names = FALSE
  )
} else {
  stop("❌ No TIF curves were computed (empty tif_curves).")
}

# ----------------------------------------------------------
# 3) Within-dataset diagnostics: how much does training
#    change the information geometry?
#    (RMSD + peak ratio; saved as CSV)
# ----------------------------------------------------------
tif_diag <- bind_rows(lapply(datasets_all, function(ds) {
  
  key_zs <- paste(ds, "zeroshot", sep = "_")
  key_tr <- paste(ds, "trained",  sep = "_")
  
  if (!(key_zs %in% names(tif_curves) && key_tr %in% names(tif_curves))) {
    message("⚠️ Missing TIF for zeroshot or trained in ", ds, "; skipping diagnostics row.")
    return(NULL)
  }
  
  df <- bind_rows(
    tif_curves[[key_zs]] %>% mutate(Curve = "Zero-shot"),
    tif_curves[[key_tr]] %>% mutate(Curve = "Trained")
  ) %>% arrange(theta, Curve)
  
  tif_zs <- df %>% filter(Curve == "Zero-shot") %>% pull(TIF)
  tif_tr <- df %>% filter(Curve == "Trained")  %>% pull(TIF)
  
  data.frame(
    Dataset          = ds,
    tif_rmsd_within  = sqrt(mean((tif_tr - tif_zs)^2, na.rm = TRUE)),
    peak_ratio       = max(tif_tr, na.rm = TRUE) / max(tif_zs, na.rm = TRUE)
  )
}))

if (nrow(tif_diag) > 0) {
  write.csv(
    tif_diag,
    file = file.path(tif_root, "tif_within_dataset_diagnostics.csv"),
    row.names = FALSE
  )
  print(tif_diag)
}

# ----------------------------------------------------------
# 4) Thesis-ready figure:
#    ONE 1×3 grid (ImageNet-C, ImageNet-Sketch, CIFAR-100)
#    Shared axis limits via log1p(TIF)
#    Facet strip labels include (a)(b)(c) ABOVE dataset name
# ----------------------------------------------------------
out_dir <- file.path(tif_root, "within_dataset_plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# desired thesis display order
ds_plot_order <- c("ImageNet-C", "ImageNet-Sketch", "CIFAR-100")

# facet strip labels (a/b/c above dataset name)
strip_labels <- c(
  "ImageNet-C"      = "(a) ImageNet-C",
  "ImageNet-Sketch" = "(b) ImageNet-Sketch",
  "CIFAR-100"       = "(c) CIFAR-100"
)

df_plot <- bind_rows(lapply(ds_plot_order, function(ds_display) {
  
  # map display name -> keys used in tif_curves
  ds_key <- dplyr::case_when(
    ds_display == "ImageNet-Sketch" ~ "Sketch",
    ds_display == "CIFAR-100"       ~ "CIFAR100",
    TRUE                            ~ ds_display
  )
  
  key_zs <- paste(ds_key, "zeroshot", sep = "_")
  key_tr <- paste(ds_key, "trained",  sep = "_")
  
  if (!(key_zs %in% names(tif_curves) && key_tr %in% names(tif_curves))) {
    message("⚠️ Missing TIF for zeroshot or trained in ", ds_key, "; skipping panel.")
    return(NULL)
  }
  
  bind_rows(
    tif_curves[[key_zs]] %>% mutate(Curve = "Zero-shot"),
    tif_curves[[key_tr]] %>% mutate(Curve = "Trained")
  ) %>%
    mutate(
      Dataset = ds_display,
      Curve   = factor(Curve, levels = c("Zero-shot", "Trained"))
    )
}))

if (nrow(df_plot) == 0) stop("❌ No datasets available to plot in the requested order.")

df_plot <- df_plot %>%
  mutate(Dataset = factor(Dataset, levels = ds_plot_order))

p_grid <- ggplot(df_plot, aes(x = theta, y = TIF, color = Curve)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(
    ~ Dataset, nrow = 1, scales = "fixed",
    labeller = as_labeller(strip_labels)
  ) +
  scale_y_continuous(trans = "log1p") +
  labs(
    x = expression(theta),
    y = "Test information (log1p scale)",
    color = NULL
  ) +
  theme_classic(base_size = 12) +
  theme(
    legend.position  = "top",
    strip.background = element_rect(fill = "grey95", color = NA),
    strip.text       = element_text(face = "bold"),
    axis.title       = element_text(face = "bold"),
    panel.grid       = element_blank()
  )

print(p_grid)

# Save figure
ggsave(
  filename = file.path(out_dir, "TIF_grid_zeroshot_vs_trained_log1p.pdf"),
  plot     = p_grid,
  width    = 10.5, height = 3.8,
  device   = cairo_pdf
)

ggsave(
  filename = file.path(out_dir, "TIF_grid_zeroshot_vs_trained_log1p.png"),
  plot     = p_grid,
  width    = 10.5, height = 3.6,
  dpi      = 400
)

cat("✅ Saved TIF grid (PDF+PNG) in:\n   ", out_dir, "\n", sep = "")



