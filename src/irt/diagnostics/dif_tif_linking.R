# ============================================================
# dif_tif_linking.R
# Differential Item Functioning (DIF), Linking, and Test Information Functions (TIF)
# Repo-safe version for DIF, linking, and TIF comparison
# ============================================================

rm(list = ls())

library(dplyr)
library(ggplot2)
library(readr)
library(tibble)
library(stringr)
library(tidyr)
library(viridis)
library(yaml)

# -----------------------------
# Load configuration
# -----------------------------
config <- yaml.load_file("config/config.yml")

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."
pred_root   <- file.path(repo_root, config$paths$predictions)  # Use config path for predictions
irt_root    <- file.path(repo_root, config$paths$theta_estimates)  # Use config path for theta estimates
dif_root    <- file.path(repo_root, config$paths$analysis, "irt", "dif_within_dataset")  # Use config path for DIF
link_root   <- file.path(repo_root, config$paths$analysis, "irt", "irt_linking")  # Use config path for linking
tif_root    <- file.path(repo_root, config$paths$analysis, "irt", "test_information")  # Use config path for TIF

# Create necessary directories if not exist
dir.create(dif_root,  recursive = TRUE, showWarnings = FALSE)
dir.create(link_root, recursive = TRUE, showWarnings = FALSE)
dir.create(tif_root,  recursive = TRUE, showWarnings = FALSE)

# Define datasets and modes
datasets_dif   <- c("Sketch", "ImageNet-C", "CIFAR100")
datasets_all   <- c("ImageNet", "Sketch", "ImageNet-C", "CIFAR100")
modes_to_check <- c("zeroshot", "trained")

# ==========================================================
# 1️⃣ Mapping tables (ImageNet ↔ CIFAR superclasses)
#    (used in DIF part for superclass summaries)
# ==========================================================
map_df <- read.csv(config$datasets$imagenet_to_cifar_map, stringsAsFactors = FALSE) %>%
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
# 2️⃣ Helper functions for IRT processing
# ==========================================================

extract_class <- function(item) {
  gsub("/.*", "", item)
}

load_item_params <- function(dataset, mode) {
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

  # Ensure column names are as expected
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

  ip$Dataset <- dataset
  ip$Mode    <- mode
  return(ip)
}

# ---------------------------------------------------------
# Part A: DIF (zeroshot vs trained) within dataset
# ---------------------------------------------------------
compute_dif_for_dataset <- function(dataset) {
  message("=== DIF analysis (zeroshot vs trained) for: ", dataset, " ===")
  
  ip_zs <- load_item_params(dataset, "zeroshot")
  ip_tr <- load_item_params(dataset, "trained")
  
  if (is.null(ip_zs) || is.null(ip_tr)) {
    message("⚠️ Missing parameters for ", dataset, " — skipping DIF.")
    return(NULL)
  }
  
  # Overlapping items only
  common_items <- intersect(ip_zs$Item, ip_tr$Item)
  n_common     <- length(common_items)
  message(sprintf("Found %d common items between zeroshot and trained for %s.",
                  n_common, dataset))
  
  if (n_common == 0) {
    message("❌ No overlapping items; cannot compute DIF for ", dataset)
    return(NULL)
  }
  
  ip_zs <- ip_zs %>% filter(Item %in% common_items)
  ip_tr <- ip_tr %>% filter(Item %in% common_items)
  
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

  dif_items <- dif_items %>%
    mutate(
      sig_sdif_b = abs(delta_b) >= 2,
      sig_sdif_a = abs(delta_a) >= 2
    )
  
  write.csv(
    dif_items,
    file = file.path(dif_root, paste0("dif_items_", dataset, ".csv")),
    row.names = FALSE
  )

  # Plot DIF results
  p_db <- ggplot(dif_items, aes(x = delta_b)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
    labs(
      title = paste("DIF Distribution for", dataset),
      x = "Delta b",
      y = "Frequency"
    ) +
    theme_minimal()

  ggsave(file.path(dif_root, paste0("hist_delta_b_", dataset, ".png")), p_db, width = 7, height = 4)
}

# ==========================================================
# Part B: IRT Linking / Test Equating to ImageNet (zeroshot)
# ==========================================================
link_irt_to_imagenet <- function(dataset, mode) {
  message("=== IRT Linking to ImageNet (zeroshot) for: ", dataset, " ===")
  
  ip_ref <- load_item_params("ImageNet", "zeroshot")
  ip_tgt <- load_item_params(dataset, mode)
  
  if (is.null(ip_ref) || is.null(ip_tgt)) {
    message("⚠️ Missing item params for ", dataset, " or ImageNet zeroshot — skipping linking.")
    return(NULL)
  }
  
  # Linking procedure (scale reference and target difficulty)
  mean_b_ref <- mean(ip_ref$b, na.rm = TRUE)
  sd_b_ref   <- sd(ip_ref$b,   na.rm = TRUE)
  
  mean_b_tgt <- mean(ip_tgt$b, na.rm = TRUE)
  sd_b_tgt   <- sd(ip_tgt$b,   na.rm = TRUE)
  
  A <- sd_b_ref / sd_b_tgt
  B <- mean_b_ref - A * mean_b_tgt
  
  b_linked <- A * ip_tgt$b + B
  
  # Save results
  ip_tgt$b_linked <- b_linked
  
  write.csv(
    ip_tgt,
    file = file.path(link_root, paste0(dataset, "_", mode, "_linked.csv")),
    row.names = FALSE
  )
  
  message("🔗 Linking saved to: ", file.path(link_root, paste0(dataset, "_", mode, "_linked.csv")))
}

# ==========================================================
# Part C: Test Information Function (TIF) Comparison
# ==========================================================
compute_tif_comparison <- function(dataset) {
  message("=== TIF Comparison for: ", dataset, " ===")
  
  ip_zs <- load_item_params(dataset, "zeroshot")
  ip_tr <- load_item_params(dataset, "trained")
  
  if (is.null(ip_zs) || is.null(ip_tr)) {
    message("⚠️ Missing parameters for ", dataset, " — skipping TIF comparison.")
    return(NULL)
  }
  
  theta_grid <- seq(-4, 4, length.out = 201)
  
  tif_zs <- tif_from_items(theta_grid, ip_zs$a, ip_zs$b)
  tif_tr <- tif_from_items(theta_grid, ip_tr$a, ip_tr$b)
  
  # Save TIF results
  tif_df <- data.frame(
    theta = theta_grid,
    TIF_zs = tif_zs,
    TIF_tr = tif_tr
  )
  
  write.csv(
    tif_df,
    file = file.path(tif_root, paste0(dataset, "_tif_comparison.csv")),
    row.names = FALSE
  )
  
  message("📈 TIF comparison saved to: ", file.path(tif_root, paste0(dataset, "_tif_comparison.csv")))
}

# ==========================================================
# Running the analysis for all datasets and modes
# ==========================================================
datasets <- c("ImageNet", "ImageNet-C", "Sketch", "CIFAR100")
for (dataset in datasets) {
  for (mode in modes_to_check) {
    run_dif_analysis(dataset, mode)
    link_irt_to_imagenet(dataset, mode)
    compute_tif_comparison(dataset)
  }
}

cat("DIF, Linking, and TIF analysis completed.\n")
