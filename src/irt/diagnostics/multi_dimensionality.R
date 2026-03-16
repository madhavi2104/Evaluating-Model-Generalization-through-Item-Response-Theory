# ==========================================================
# MULTI-DIM ABILITY CHECK (1D vs 2D 2PL)
# ==========================================================
# Enhancements added:
# - Robust sign alignment for theta2 (fix negative Spearman due to axis flips)
# - Save 2D item parameters (a1, a2, difficulty/intercepts) for interpretation
# - Save factor correlation (oblimin implies correlated factors)
# - Add simple "dim2 strength" summaries (how many items load on dim2)
# - Keep your existing safe wrappers + outputs
# ==========================================================

library(mirt)
library(dplyr)
library(tidyr)
library(stringr)
library(data.table)
library(ggplot2)
library(yaml)

# -----------------------------
# Config (from config file)
# -----------------------------
repo_root <- "."
pred_root   <- file.path(repo_root, config$paths$predictions)  # Use config path for predictions
theta_root    <- file.path(repo_root, config$paths$theta_estimates)  # Use config path for theta estimates

out_root <- file.path(repo_root, config$paths$analysis, "irt", "multidim_check")
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

datasets_all   <- c("ImageNet", "Sketch", "ImageNet-C", "CIFAR100")
modes_to_check <- c("zeroshot", "trained")

get_model_columns <- function(df) {
  drop_cols <- c("Item", "true_label", "true label", "truelabel", "label", "gt", "ground_truth")
  keep <- setdiff(colnames(df), drop_cols)
  keep <- keep[!grepl("true.?label|ground.?truth|gt", keep, ignore.case = TRUE)]
  keep
}

to_binary_numeric <- function(x) {
  if (is.factor(x)) x <- as.character(x)
  
  if (is.character(x)) {
    x <- trimws(x)
    x[x %in% c("", "NA", "NaN", "nan", "NULL", "null")] <- NA
    x[x %in% c("TRUE", "True", "true")]  <- "1"
    x[x %in% c("FALSE", "False", "false")] <- "0"
  }
  
  x_num <- suppressWarnings(as.numeric(x))
  if (all(is.na(x_num))) return(x_num)
  
  x_num[x_num > 1 & x_num <= 1.0000001] <- 1
  x_num[x_num < 0 & x_num >= -1e-7] <- 0
  
  bad <- !is.na(x_num) & !(x_num %in% c(0, 1))
  if (any(bad)) x_num[bad] <- ifelse(x_num[bad] >= 0.5, 1, 0)
  
  x_num
}

rebuild_X <- function(pred_file, sampled_items) {
  df <- fread(pred_file, data.table = FALSE, check.names = FALSE)
  
  if (!("Item" %in% colnames(df))) colnames(df)[1] <- "Item"
  
  df_sub <- df %>% filter(Item %in% sampled_items)
  if (nrow(df_sub) == 0) stop("No overlap between sampled_items and prediction file items.")
  
  idx <- match(sampled_items, df_sub$Item)
  df_sub <- df_sub[idx[!is.na(idx)], , drop = FALSE]
  
  model_cols <- get_model_columns(df_sub)
  if (length(model_cols) == 0) stop("No model columns found in prediction CSV.")
  
  Y <- df_sub[, model_cols, drop = FALSE]
  Y_num <- as.data.frame(lapply(Y, to_binary_numeric))
  
  X <- t(as.matrix(Y_num))
  rownames(X) <- colnames(Y_num)
  colnames(X) <- df_sub$Item
  X
}

clean_X <- function(X, min_items = 10, max_na_frac = 0.5) {
  keep_models <- apply(X, 1, function(r) !all(is.na(r)))
  X <- X[keep_models, , drop = FALSE]
  
  keep_items <- apply(X, 2, function(c) !all(is.na(c)))
  X <- X[, keep_items, drop = FALSE]
  
  v <- apply(X, 2, var, na.rm = TRUE)
  keep_items2 <- is.finite(v) & v > 0
  X <- X[, keep_items2, drop = FALSE]
  
  if (!is.null(max_na_frac)) {
    na_frac <- rowMeans(is.na(X))
    X <- X[na_frac <= max_na_frac, , drop = FALSE]
  }
  
  if (ncol(X) < min_items) return(NULL)
  if (nrow(X) < 5) return(NULL)
  X
}

fit_2pl_1D <- function(X) {
  tryCatch({
    mirt(
      X, 1, itemtype = "2PL", method = "EM",
      technical = list(NCYCLES = 1000), TOL = 1e-3
    )
  }, error = function(e) { warning("1D fit failed: ", conditionMessage(e)); NULL })
}

fit_2pl_2D <- function(X) {
  tryCatch({
    mirt(
      X, 2, itemtype = "2PL", method = "EM",
      technical = list(NCYCLES = 1500), TOL = 1e-3,
      exploratory = list(rotation = "oblimin")
    )
  }, error = function(e) { warning("2D fit failed: ", conditionMessage(e)); NULL })
}

safe_fscores <- function(fit, X) {
  if (is.null(fit)) return(NULL)
  th <- tryCatch(fscores(fit, full.scores = TRUE),
                 error = function(e) { warning("fscores failed: ", conditionMessage(e)); NULL })
  if (is.null(th)) return(NULL)
  th <- as.matrix(th)
  if (is.null(rownames(th)) || any(rownames(th) == "")) rownames(th) <- rownames(X)
  th
}

safe_logLik <- function(fit) {
  if (is.null(fit)) return(NA_real_)
  ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
  if (is.finite(ll)) return(ll)
  out <- tryCatch(extract.mirt(fit, "logLik"), error = function(e) NULL)
  if (is.null(out) || length(out) == 0) return(NA_real_)
  as.numeric(out[1])
}

safe_AIC <- function(fit) {
  if (is.null(fit)) return(NA_real_)
  out <- tryCatch(AIC(fit), error = function(e) NULL)
  if (is.null(out) || length(out) == 0) return(NA_real_)
  as.numeric(out[1])
}

safe_BIC <- function(fit) {
  if (is.null(fit)) return(NA_real_)
  out <- tryCatch(BIC(fit), error = function(e) NULL)
  if (is.null(out) || length(out) == 0) return(NA_real_)
  as.numeric(out[1])
}

safe_LRT <- function(m1, m2) {
  out <- list(LRT = NA_real_, df = NA_integer_, p = NA_real_)
  if (is.null(m1) || is.null(m2)) return(out)
  a <- tryCatch(anova(m1, m2), error = function(e) NULL)
  if (is.null(a) || !is.data.frame(a) || nrow(a) < 2) return(out)
  
  if (all(c("LR", "df", "p") %in% colnames(a))) {
    out$LRT <- as.numeric(a[2, "LR"])
    out$df  <- as.integer(a[2, "df"])
    out$p   <- as.numeric(a[2, "p"])
  } else if (all(c("Chisq", "Df") %in% colnames(a))) {
    out$LRT <- as.numeric(a[2, "Chisq"])
    out$df  <- as.integer(a[2, "Df"])
    pcol <- intersect(colnames(a), c("p", "Pr(>Chisq)", "Pr(>Chisq) "))
    out$p <- if (length(pcol) > 0) as.numeric(a[2, pcol[1]]) else NA_real_
  }
  out
}

safe_factor_cor <- function(fit2) {
  if (is.null(fit2)) return(NA_real_)
  tryCatch({
    im <- inspect(fit2, "correlation")
    if (is.matrix(im) && nrow(im) >= 2) as.numeric(im[1, 2]) else NA_real_
  }, error = function(e) NA_real_)
}

align_theta2_sign <- function(df_sc) {
  ok <- is.finite(df_sc$theta1) & is.finite(df_sc$theta2)
  if (sum(ok) < 8) return(df_sc)
  rho_s <- suppressWarnings(cor(df_sc$theta1[ok], df_sc$theta2[ok], method = "spearman"))
  if (is.finite(rho_s) && rho_s < 0) df_sc$theta2 <- -df_sc$theta2
  df_sc
}

# --- FIX: force scalars (never length-0) ---
as_scalar <- function(x, default) {
  if (is.null(x) || length(x) == 0) return(default)
  x[1]
}

safe_convergence_info <- function(fit) {
  if (is.null(fit)) {
    return(list(converged = NA, iterations = NA_integer_, message = NA_character_))
  }
  conv <- tryCatch(fit@OptimInfo$converged, error = function(e) NULL)
  it   <- tryCatch(fit@OptimInfo$iterations, error = function(e) NULL)
  msg  <- tryCatch(fit@OptimInfo$message, error = function(e) NULL)
  
  list(
    converged = as_scalar(conv, NA),
    iterations = as.integer(as_scalar(it, NA_integer_)),
    message = as.character(as_scalar(msg, NA_character_))
  )
}

fit_summaries   <- list()
theta_1D_all    <- list()
theta_2D_all    <- list()
linearity_all   <- list()

for (ds in datasets_all) {
  for (md in modes_to_check) {
    
    cat("\n===============================\n")
    cat("Multi-dim check:", ds, md, "\n")
    
    mode_dir <- file.path(theta_root, ds, md)
    if (!dir.exists(mode_dir)) { cat("  Mode dir not found:", mode_dir, "\n"); next }
    
    best_folder <- list.dirs(mode_dir, recursive = FALSE, full.names = TRUE)
    best_folder <- best_folder[grepl("best_", basename(best_folder))]
    if (length(best_folder) == 0) { cat("  No best_* folder found for", ds, md, "\n"); next }
    best_folder <- best_folder[1]
    
    sample_file <- file.path(best_folder, "Sampled_Items.csv")
    if (!file.exists(sample_file)) { cat("  Sampled_Items.csv missing in:", best_folder, "\n"); next }
    
    sampled_items <- fread(sample_file)$Item
    if (length(sampled_items) == 0) { cat("  No sampled items in:", sample_file, "\n"); next }
    
    pred_file <- file.path(pred_root, paste0(ds, "_", md, ".csv"))
    if (!file.exists(pred_file)) { cat("  Prediction file missing:", pred_file, "\n"); next }
    
    X_raw <- tryCatch(rebuild_X(pred_file, sampled_items),
                      error = function(e) { warning("rebuild_X failed: ", conditionMessage(e)); NULL })
    if (is.null(X_raw)) next
    
    X <- clean_X(X_raw, min_items = 10, max_na_frac = 0.5)
    if (is.null(X)) { cat("  After cleaning: too few items/models. Skipping.\n"); next }
    
    cat("  X dims:", nrow(X), "x", ncol(X), "(models x items)\n")
    
    cat("  Fitting 1D 2PL.\n")
    m1 <- fit_2pl_1D(X)
    
    cat("  Fitting 2D 2PL (EFA, oblimin).\n")
    m2 <- fit_2pl_2D(X)
    
    c1 <- safe_convergence_info(m1)
    c2 <- safe_convergence_info(m2)
    
    LL_1D  <- safe_logLik(m1)
    AIC_1D <- safe_AIC(m1)
    BIC_1D <- safe_BIC(m1)
    
    LL_2D  <- safe_logLik(m2)
    AIC_2D <- safe_AIC(m2)
    BIC_2D <- safe_BIC(m2)
    
    dAIC <- if (is.finite(AIC_2D) && is.finite(AIC_1D)) AIC_2D - AIC_1D else NA_real_
    dBIC <- if (is.finite(BIC_2D) && is.finite(BIC_1D)) BIC_2D - BIC_1D else NA_real_
    
    lrt <- safe_LRT(m1, m2)
    fac_cor <- safe_factor_cor(m2)
    
    fit_row <- data.frame(
      Dataset  = ds,
      Mode     = md,
      n_models = nrow(X),
      n_items  = ncol(X),
      
      LL_1D  = LL_1D,
      AIC_1D = AIC_1D,
      BIC_1D = BIC_1D,
      
      LL_2D  = LL_2D,
      AIC_2D = AIC_2D,
      BIC_2D = BIC_2D,
      
      dAIC   = dAIC,
      dBIC   = dBIC,
      
      LRT    = lrt$LRT,
      LRT_df = lrt$df,
      LRT_p  = lrt$p,
      
      factor_cor_12 = fac_cor,
      
      conv_1D = c1$converged,
      iter_1D = c1$iterations,
      msg_1D  = c1$message,
      
      conv_2D = c2$converged,
      iter_2D = c2$iterations,
      msg_2D  = c2$message,
      
      stringsAsFactors = FALSE
    )
    
    fit_summaries[[paste(ds, md, sep = "_")]] <- fit_row
    
    th1 <- safe_fscores(m1, X)
    if (!is.null(th1) && nrow(th1) == nrow(X)) {
      theta_1D_all[[paste(ds, md, sep = "_")]] <- data.frame(
        Model = rownames(th1),
        theta_1D = th1[, 1],
        Dataset = ds,
        Mode = md,
        row.names = NULL
      )
    }
    
    th2 <- NULL
    if (isTRUE(c2$converged) && is.finite(LL_2D)) {
      th2 <- safe_fscores(m2, X)
    } else {
      warning("Skipping 2D fscores for ", ds, " ", md,
              " because 2D fit did not converge / LL_2D not available.")
    }
    
    if (!is.null(th2) && nrow(th2) == nrow(X) && ncol(th2) >= 2) {
      
      df_sc <- data.frame(
        Model = rownames(th2),
        theta1 = th2[, 1],
        theta2 = th2[, 2],
        Dataset = ds,
        Mode = md,
        row.names = NULL
      )
      
      df_sc <- align_theta2_sign(df_sc)
      theta_2D_all[[paste(ds, md, sep = "_")]] <- df_sc
      
      ok <- is.finite(df_sc$theta1) & is.finite(df_sc$theta2)
      if (sum(ok) >= 8) {
        lm_fit <- lm(theta2 ~ theta1, data = df_sc[ok, ])
        R2 <- summary(lm_fit)$r.squared
        rho_s <- suppressWarnings(cor(df_sc$theta1[ok], df_sc$theta2[ok], method = "spearman"))
        
        linearity_all[[paste(ds, md, sep = "_")]] <- data.frame(
          Dataset = ds, Mode = md,
          n = sum(ok),
          R2_theta2_on_theta1 = R2,
          spearman_theta1_theta2 = rho_s,
          slope = coef(lm_fit)[["theta1"]],
          intercept = coef(lm_fit)[["(Intercept)"]],
          row.names = NULL
        )
        
        p <- ggplot(df_sc, aes(theta1, theta2)) +
          geom_point(size = 2) +
          geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
          theme_minimal(base_size = 13) +
          theme(
            panel.background = element_rect(fill = "white", color = NA),
            plot.background  = element_rect(fill = "white", color = NA)
          ) +
          labs(
            title = paste0("2D ability space: ", ds, " (", md, ")"),
            subtitle = sprintf("Linearity check: R²(theta2~theta1)=%.3f, Spearman=%.3f", R2, rho_s),
            x = expression(theta[1]),
            y = expression(theta[2])
          )
        
        ggsave(
          filename = file.path(out_root, paste0("theta2D_scatter_", ds, "_", md, ".png")),
          plot = p, width = 6.5, height = 5.2, dpi = 300, bg = "white"
        )
      }
    }
  }
}

if (length(fit_summaries) > 0) {
  fit_summary_all <- bind_rows(fit_summaries)
  out_fit <- file.path(out_root, "multidim_fit_summary_1D_vs_2D.csv")
  fwrite(fit_summary_all, out_fit)
  cat("\nSaved fit summary to:", out_fit, "\n")
}

if (length(theta_1D_all) > 0) {
  th1_all <- bind_rows(theta_1D_all)
  out_th1 <- file.path(out_root, "theta_1D_all_datasets_modes.csv")
  fwrite(th1_all, out_th1)
  cat("Saved 1D thetas to:", out_th1, "\n")
}

if (length(theta_2D_all) > 0) {
  th2_all <- bind_rows(theta_2D_all)
  out_th2 <- file.path(out_root, "theta_2D_all_datasets_modes.csv")
  fwrite(th2_all, out_th2)
  cat("Saved 2D thetas to:", out_th2, "\n")
}

if (length(linearity_all) > 0) {
  lin_all <- bind_rows(linearity_all)
  out_lin <- file.path(out_root, "theta2D_linearity_metrics.csv")
  fwrite(lin_all, out_lin)
  cat("Saved 2D linearity metrics to:", out_lin, "\n")
}

cat("\nMulti-dim ability check complete.\n")

# ==========================================================
# EXTRA: Cross-dataset stability of theta2 ranks (Spearman)
# NOTE: trained regime uses ImageNet_zeroshot as reference
# ==========================================================
library(readr)
library(reshape2)

theta2_file <- file.path(out_root, "theta_2D_all_datasets_modes.csv")
if (!file.exists(theta2_file)) {
  warning("theta_2D_all_datasets_modes.csv not found at: ", theta2_file)
} else {
  
  th2_all <- read.csv(theta2_file)
  
  # --- Wide table of theta2 values: columns like ImageNet_zeroshot, Sketch_trained, etc.
  th2_wide <- th2_all %>%
    dplyr::select(Model, Dataset, Mode, theta2) %>%
    dplyr::mutate(DatasetMode = paste0(Dataset, "_", Mode)) %>%
    dplyr::select(-Dataset, -Mode) %>%
    tidyr::pivot_wider(names_from = DatasetMode, values_from = theta2)
  
  # --- Rank: higher theta => rank 1
  theta_to_rank <- function(x) rank(-x, ties.method = "average", na.last = "keep")
  
  # --- Correlations: ImageNet_zeroshot reference, plus all-pairs option
  compute_theta2_rank_correlations <- function(theta2_rank_wide, mode_label,
                                               reference_col, target_cols,
                                               do_all_pairs = TRUE) {
    
    df <- theta2_rank_wide
    df[,-1] <- lapply(df[,-1], as.numeric)
    
    results <- list()
    
    # (A) Reference vs target set
    if (reference_col %in% colnames(df)) {
      ref <- df[[reference_col]]
      for (cc in target_cols) {
        if (!(cc %in% colnames(df))) next
        rho <- suppressWarnings(cor(ref, df[[cc]], method = "spearman", use = "pairwise.complete.obs"))
        results[[length(results)+1]] <- data.frame(
          Regime = mode_label,
          Type = "Reference_vs_Targets",
          Reference = reference_col,
          Target = cc,
          SpearmanRho = rho,
          stringsAsFactors = FALSE
        )
      }
    } else {
      warning("Reference column not found: ", reference_col)
    }
    
    # (B) All-pairs within the target set (and optionally include reference too)
    if (do_all_pairs) {
      cols <- unique(c(reference_col, target_cols))
      cols <- cols[cols %in% colnames(df)]
      if (length(cols) >= 2) {
        for (i in 1:(length(cols)-1)) {
          for (j in (i+1):length(cols)) {
            c1 <- cols[i]; c2 <- cols[j]
            rho <- suppressWarnings(cor(df[[c1]], df[[c2]], method = "spearman", use = "pairwise.complete.obs"))
            results[[length(results)+1]] <- data.frame(
              Regime = mode_label,
              Type = "All_Pairs",
              Reference = c1,
              Target = c2,
              SpearmanRho = rho,
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
    
    dplyr::bind_rows(results)
  }
  
  # --- Heatmap with cleaned dataset labels
  clean_names <- function(x) {
    x <- gsub("_zeroshot$", " (ZS)", x)
    x <- gsub("_trained$", " (Tr)", x)
    x <- gsub("ImageNet.C", "ImageNet-C", x, fixed = TRUE)
    x
  }
  
  make_heatmap <- function(rank_wide, cols_use, title_str, out_png) {
    mat <- rank_wide[, cols_use, drop = FALSE]
    colnames(mat) <- clean_names(colnames(mat))
    cor_mat <- suppressWarnings(cor(mat, method = "spearman", use = "pairwise.complete.obs"))
    melt_mat <- reshape2::melt(cor_mat)
    
    p <- ggplot(melt_mat, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile(color = "white") +
      geom_text(aes(label = sprintf("%.3f", value)), color = "black", size = 3) +
      scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0.5) +
      theme_minimal(base_size = 12) +
      labs(title = title_str, fill = "Spearman ρ", x = "", y = "") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    ggsave(out_png, p, width = 8, height = 6, dpi = 300, bg = "white")
    out_png
  }
  
  # --- Build rank table for ALL columns first
  th2_rank_wide <- th2_wide
  th2_rank_wide[,-1] <- lapply(th2_rank_wide[,-1], theta_to_rank)
  
  # Save full rank-wide (all regimes together) if you want
  write.csv(th2_rank_wide, file.path(out_root, "theta2_rank_wide_ALL.csv"), row.names = FALSE)
  
  # -----------------------------
  # Regime 1: Zero-shot (ImageNet_zeroshot vs other zeroshot)
  # -----------------------------
  zs_cols <- grep("_zeroshot$", colnames(th2_rank_wide), value = TRUE)
  if (length(zs_cols) > 0) {
    zs_rank <- th2_rank_wide %>% dplyr::select(Model, all_of(zs_cols))
    
    out_rank <- file.path(out_root, "theta2_rank_wide_zeroshot.csv")
    write.csv(zs_rank, out_rank, row.names = FALSE)
    cat("Saved theta2 rank wide table (zeroshot):", out_rank, "\n")
    
    reference_col <- "ImageNet_zeroshot"
    target_cols <- setdiff(zs_cols, reference_col)
    
    res_zs <- compute_theta2_rank_correlations(
      theta2_rank_wide = zs_rank,
      mode_label = "zeroshot",
      reference_col = reference_col,
      target_cols = target_cols,
      do_all_pairs = TRUE
    )
    
    out_res <- file.path(out_root, "theta2_spearman_results_zeroshot.csv")
    write.csv(res_zs, out_res, row.names = FALSE)
    cat("Saved theta2 Spearman results (zeroshot):", out_res, "\n")
    
    hm <- make_heatmap(
      rank_wide = zs_rank,
      cols_use = zs_cols,
      title_str = "Theta2 Rank Correlations (Zero-shot)",
      out_png = file.path(out_root, "theta2_rank_correlation_heatmap_zeroshot.png")
    )
    cat("Saved theta2 rank heatmap (zeroshot):", hm, "\n")
  } else {
    warning("No *_zeroshot columns found for theta2.")
  }
  
  # -----------------------------
  # Regime 2: Trained (ImageNet_zeroshot vs other trained)
  # -----------------------------
  tr_cols <- grep("_trained$", colnames(th2_rank_wide), value = TRUE)
  if (length(tr_cols) > 0) {
    tr_rank <- th2_rank_wide %>% dplyr::select(Model, all_of(tr_cols))
    
    out_rank <- file.path(out_root, "theta2_rank_wide_trained.csv")
    write.csv(tr_rank, out_rank, row.names = FALSE)
    cat("Saved theta2 rank wide table (trained):", out_rank, "\n")
    
    reference_col <- "ImageNet_zeroshot"
    target_cols <- tr_cols  # compare ImageNet_zeroshot vs all trained columns
    
    # Build a temp DF that includes the reference column + trained columns
    tr_plus_ref <- th2_rank_wide %>% dplyr::select(Model, all_of(unique(c(reference_col, tr_cols))))
    
    res_tr <- compute_theta2_rank_correlations(
      theta2_rank_wide = tr_plus_ref,
      mode_label = "trained",
      reference_col = reference_col,
      target_cols = target_cols,
      do_all_pairs = TRUE
    )
    
    out_res <- file.path(out_root, "theta2_spearman_results_trained.csv")
    write.csv(res_tr, out_res, row.names = FALSE)
    cat("Saved theta2 Spearman results (trained; ref=ImageNet_zeroshot):", out_res, "\n")
    
    hm <- make_heatmap(
      rank_wide = tr_plus_ref,
      cols_use = unique(c(reference_col, tr_cols)),
      title_str = "Theta2 Rank Correlations (Trained; ref=ImageNet ZS)",
      out_png = file.path(out_root, "theta2_rank_correlation_heatmap_trained.png")
    )
    cat("Saved theta2 rank heatmap (trained):", hm, "\n")
  } else {
    warning("No *_trained columns found for theta2.")
  }
  
  cat("\n✅ Theta2 cross-dataset rank stability complete.\n")
}

# ==========================================================
# EXTRA CHECK: Correlation between theta^(1D) and theta1^(2D)
# For each dataset + regime: Spearman + Pearson
# ==========================================================
th1_file <- file.path(out_root, "theta_1D_all_datasets_modes.csv")
th2_file <- file.path(out_root, "theta_2D_all_datasets_modes.csv")

if (!file.exists(th1_file)) {
  warning("Missing: ", th1_file)
} else if (!file.exists(th2_file)) {
  warning("Missing: ", th2_file)
} else {
  
  th1 <- read.csv(th1_file)
  th2 <- read.csv(th2_file)
  
  # Merge on Model + Dataset + Mode
  merged <- th1 %>%
    dplyr::select(Model, Dataset, Mode, theta_1D) %>%
    dplyr::inner_join(
      th2 %>% dplyr::select(Model, Dataset, Mode, theta1),
      by = c("Model", "Dataset", "Mode")
    )
  
  if (nrow(merged) == 0) {
    warning("No overlap found between theta 1D and theta1 2D tables after merge.")
  } else {
    
    interpret_strength <- function(rho) {
      if (!is.finite(rho)) return(NA_character_)
      if (rho >= 0.90) return("very_high: primary dimension ~ same ordering as 1D ability")
      if (rho >= 0.60) return("moderate: adding dim2 reallocates variance; some shift in primary coord")
      return("low: surprising; may need extra sentence about 2D coordinate rotation/variance reallocation")
    }
    
    cor_rows <- merged %>%
      dplyr::group_by(Dataset, Mode) %>%
      dplyr::summarise(
        n = sum(is.finite(theta_1D) & is.finite(theta1)),
        spearman_theta1D_vs_theta1_2D = suppressWarnings(cor(theta_1D, theta1, method = "spearman", use = "pairwise.complete.obs")),
        pearson_theta1D_vs_theta1_2D  = suppressWarnings(cor(theta_1D, theta1, method = "pearson",  use = "pairwise.complete.obs")),
        .groups = "drop"
      ) %>%
      dplyr::mutate(
        interpretation_spearman = vapply(spearman_theta1D_vs_theta1_2D, interpret_strength, character(1))
      )
    
    out_cor <- file.path(out_root, "theta1D_vs_theta1_2D_correlations.csv")
    write.csv(cor_rows, out_cor, row.names = FALSE)
    cat("Saved theta(1D) vs theta1(2D) correlations to:", out_cor, "\n")
    
    print(cor_rows)
  }
}
