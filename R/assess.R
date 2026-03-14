#' Assess Embedding Quality
#'
#' Unified evaluation of an NMF, SVD, or arbitrary embedding matrix against
#' one or more label vectors. Computes clustering metrics (ARI, NMI, Silhouette),
#' classification metrics (KNN/LR/RF accuracy, F1, AUC via cross-validation),
#' and batch-mixing metrics (donor/batch Silhouette, kNN entropy).
#'
#' Uses GPU-accelerated kernels when available, otherwise falls back to
#' CPU-based approximate algorithms that avoid O(n^2) distance matrices:
#' \itemize{
#'   \item{Silhouette: sampled approximation (O(n * samples_per_class * C))}
#'   \item{kNN metrics: brute-force on GPU or sampled on CPU (O(n * k * n_ref))}
#'   \item{ARI/NMI: contingency table (O(n), always fast)}
#' }
#'
#' @param x An object of class \code{nmf}, \code{svd}, or a numeric matrix
#'   (samples x features). For \code{nmf} objects, uses \code{t(diag(d) \%*\% h)};
#'   for \code{svd} objects, uses \code{u \%*\% diag(d)}.
#' @param labels A factor, character, or integer vector of primary labels
#'   (e.g. cell type). Required for clustering and classification metrics.
#' @param batch Optional factor/character/integer vector of batch labels
#'   (e.g. donor_id, tissue). When provided, batch-mixing metrics are computed.
#' @param metrics Character vector specifying which metrics to compute.
#'   Options: \code{"ari"}, \code{"nmi"}, \code{"silhouette"},
#'   \code{"classification"}, \code{"batch_mixing"}, or \code{"all"} (default).
#' @param n_folds Number of cross-validation folds for classification (default 5).
#' @param classifiers Character vector of classifiers: \code{"knn"}, \code{"lr"},
#'   \code{"rf"}, or any combination (default all three).
#' @param k_nn Number of neighbors for kNN classifier (default 15).
#' @param seed Random seed for reproducibility (default 42).
#' @param min_class_size Minimum number of samples per class. Classes smaller
#'   than this are dropped before evaluation (default 10).
#' @param sil_samples_per_class Number of reference samples per class for
#'   approximate silhouette (default 200). Higher = more accurate, slower.
#' @param batch_knn_k Number of neighbors for batch entropy (default 50).
#' @return An S3 object of class \code{nmf_assessment} containing:
#'   \describe{
#'     \item{metrics}{Named list of all computed metric values}
#'     \item{classification}{Data frame of per-classifier, per-fold results}
#'     \item{params}{List recording all parameters used}
#'   }
#' @examples
#' \donttest{
#' # Assess an NMF model
#' library(Matrix)
#' A <- abs(rsparsematrix(200, 100, 0.2))
#' model <- nmf(A, 5, seed = 1)
#' labels <- factor(sample(letters[1:4], 100, replace = TRUE))
#' result <- assess(model, labels)
#' print(result)
#'
#' # Select specific metrics
#' result <- assess(model, labels, metrics = c("nmi", "ari"))
#'
#' # Include batch assessment
#' batch <- factor(sample(c("A", "B"), 100, replace = TRUE))
#' result <- assess(model, labels, batch = batch, metrics = "all")
#' }
#' @export
assess <- function(x, labels, batch = NULL,
                   metrics = "all",
                   n_folds = 5L,
                   classifiers = c("knn", "lr", "rf"),
                   k_nn = 15L,
                   seed = 42L,
                   min_class_size = 10L,
                   sil_samples_per_class = 200L,
                   batch_knn_k = 50L) {

  # --- Extract embedding matrix ---
  embedding <- .extract_embedding(x)
  n <- nrow(embedding)
  k <- ncol(embedding)

  # --- Validate labels ---
  if (length(labels) != n)
    stop("length(labels) must equal nrow of embedding (", n, ")")
  labels <- as.factor(labels)

  # --- Parse metrics ---
  all_metrics <- c("ari", "nmi", "silhouette", "classification", "batch_mixing")
  if (length(metrics) == 1L && metrics == "all") {
    do_metrics <- if (is.null(batch)) setdiff(all_metrics, "batch_mixing") else all_metrics
  } else {
    do_metrics <- match.arg(metrics, all_metrics, several.ok = TRUE)
  }

  # Filter small classes
  class_counts <- table(labels)
  keep_classes <- names(class_counts)[class_counts >= min_class_size]
  if (length(keep_classes) < 2L)
    stop("Fewer than 2 classes with >= ", min_class_size, " samples")
  keep_mask <- labels %in% keep_classes
  if (sum(!keep_mask) > 0L) {
    message(sprintf("Dropped %d samples from %d small classes (min_class_size=%d)",
                    sum(!keep_mask), length(class_counts) - length(keep_classes),
                    min_class_size))
    embedding <- embedding[keep_mask, , drop = FALSE]
    labels <- droplevels(labels[keep_mask])
    if (!is.null(batch)) batch <- batch[keep_mask]
    n <- nrow(embedding)
  }

  labels_int <- as.integer(labels) - 1L
  n_classes <- nlevels(labels)

  # --- Check GPU availability ---
  use_gpu <- .assess_gpu_available()

  # --- Prepare batch labels ---
  batch_int <- integer(n)  # all -1 sentinel if no batch
  n_batch <- 0L
  if (!is.null(batch)) {
    batch <- as.factor(batch)
    batch_int <- as.integer(batch) - 1L
    n_batch <- nlevels(batch)
  }

  # --- GPU fast-path: compute everything in one kernel launch ---
  if (use_gpu) {
    gpu_result <- .assess_gpu(
      embedding = embedding,
      labels_int = labels_int,
      n_classes = n_classes,
      batch_int = batch_int,
      n_batch = n_batch,
      do_metrics = do_metrics,
      n_folds = n_folds,
      k_nn = k_nn,
      sil_samples_per_class = sil_samples_per_class,
      batch_knn_k = batch_knn_k,
      seed = seed
    )
    if (!is.null(gpu_result)) {
      # GPU succeeded — assemble result
      result_metrics <- list()
      clf_details <- NULL

      if ("ari" %in% do_metrics) result_metrics$ari <- gpu_result$ari
      if ("nmi" %in% do_metrics) result_metrics$nmi <- gpu_result$nmi
      if ("silhouette" %in% do_metrics) result_metrics$silhouette <- gpu_result$silhouette

      if ("classification" %in% do_metrics) {
        # GPU only provides kNN; run LR/RF on CPU if requested
        clf_results_list <- list()
        if ("knn" %in% classifiers) {
          clf_results_list[["knn"]] <- data.frame(
            classifier = "knn", fold = 0L,
            accuracy = gpu_result$knn_accuracy,
            f1 = gpu_result$knn_f1,
            precision = NA_real_, recall = NA_real_, auroc = NA_real_,
            stringsAsFactors = FALSE
          )
        }
        # LR and RF still run on CPU (reasonable scaling)
        cpu_clfs <- intersect(classifiers, c("lr", "rf"))
        if (length(cpu_clfs) > 0L) {
          cpu_clf_result <- .cv_classify(embedding, labels, n_folds = n_folds,
                                          classifiers = cpu_clfs, k_nn = k_nn,
                                          seed = seed)
          clf_results_list[["cpu"]] <- cpu_clf_result$details
        }
        clf_details <- do.call(rbind, clf_results_list)

        for (clf_name in unique(clf_details$classifier)) {
          rows <- clf_details[clf_details$classifier == clf_name, ]
          for (metric_name in c("accuracy", "f1", "precision", "recall", "auroc")) {
            key <- paste0(metric_name, "_", clf_name)
            result_metrics[[key]] <- mean(rows[[metric_name]], na.rm = TRUE)
          }
        }
        for (metric_name in c("accuracy", "f1", "precision", "recall", "auroc")) {
          result_metrics[[paste0(metric_name, "_mean")]] <-
            mean(clf_details[[metric_name]], na.rm = TRUE)
        }
      }

      if ("batch_mixing" %in% do_metrics && n_batch > 1L) {
        result_metrics$batch_silhouette <- gpu_result$batch_sil
        result_metrics$batch_knn_entropy <- gpu_result$batch_entropy
      }

      return(structure(list(
        metrics = result_metrics,
        classification = clf_details,
        params = list(
          n_samples = n, n_features = k, n_classes = n_classes,
          metrics = do_metrics, n_folds = n_folds,
          classifiers = classifiers, k_nn = k_nn, seed = seed,
          min_class_size = min_class_size,
          has_batch = !is.null(batch),
          backend = "gpu"
        )
      ), class = "nmf_assessment"))
    }
    # GPU failed — fall through to CPU
  }

  # --- CPU path (approximate, avoids O(n^2)) ---
  result_metrics <- list()
  clf_details <- NULL

  # --- Clustering metrics (ARI, NMI) ---
  if ("ari" %in% do_metrics || "nmi" %in% do_metrics) {
    set.seed(seed)
    km <- stats::kmeans(embedding, centers = n_classes, nstart = 10L, iter.max = 100L)
    pred_clusters <- km$cluster - 1L

    if ("ari" %in% do_metrics)
      result_metrics$ari <- .adjusted_rand_index(labels_int, pred_clusters)
    if ("nmi" %in% do_metrics)
      result_metrics$nmi <- .normalized_mutual_info(labels_int, pred_clusters)
  }

  # --- Silhouette (approximate, sampled) ---
  if ("silhouette" %in% do_metrics) {
    result_metrics$silhouette <- .approx_silhouette(embedding, labels_int,
                                                     spc = sil_samples_per_class,
                                                     seed = seed)
  }

  # --- Classification (stratified k-fold CV) ---
  if ("classification" %in% do_metrics) {
    clf_result <- .cv_classify(embedding, labels, n_folds = n_folds,
                               classifiers = classifiers, k_nn = k_nn,
                               seed = seed)
    clf_details <- clf_result$details

    for (clf_name in unique(clf_details$classifier)) {
      rows <- clf_details[clf_details$classifier == clf_name, ]
      for (metric_name in c("accuracy", "f1", "precision", "recall", "auroc")) {
        key <- paste0(metric_name, "_", clf_name)
        result_metrics[[key]] <- mean(rows[[metric_name]], na.rm = TRUE)
      }
    }
    for (metric_name in c("accuracy", "f1", "precision", "recall", "auroc")) {
      result_metrics[[paste0(metric_name, "_mean")]] <-
        mean(clf_details[[metric_name]], na.rm = TRUE)
    }
  }

  # --- Batch mixing ---
  if ("batch_mixing" %in% do_metrics && n_batch > 1L) {
    result_metrics$batch_silhouette <- .approx_silhouette(embedding, batch_int,
                                                           spc = sil_samples_per_class,
                                                           seed = seed + 1L)
    result_metrics$batch_knn_entropy <- .approx_batch_knn_entropy(
      embedding, batch_int, n_batch, k = batch_knn_k, seed = seed)
  }

  structure(list(
    metrics = result_metrics,
    classification = clf_details,
    params = list(
      n_samples = n, n_features = k, n_classes = n_classes,
      metrics = do_metrics, n_folds = n_folds,
      classifiers = classifiers, k_nn = k_nn, seed = seed,
      min_class_size = min_class_size,
      has_batch = !is.null(batch),
      backend = "cpu"
    )
  ), class = "nmf_assessment")
}


# ============================================================
# S3 methods
# ============================================================

#' Print method for nmf_assessment objects
#' @param x an \code{nmf_assessment} object
#' @param ... additional arguments (unused)
#' @return Invisibly returns \code{x}
#' @method print nmf_assessment
#' @export
print.nmf_assessment <- function(x, ...) {
  cat("Embedding Assessment\n")
  cat(sprintf("  %d samples, %d features, %d classes\n",
              x$params$n_samples, x$params$n_features, x$params$n_classes))

  # Clustering
  if (!is.null(x$metrics$nmi) || !is.null(x$metrics$ari)) {
    cat("\nClustering:\n")
    if (!is.null(x$metrics$nmi)) cat(sprintf("  NMI:         %.4f\n", x$metrics$nmi))
    if (!is.null(x$metrics$ari)) cat(sprintf("  ARI:         %.4f\n", x$metrics$ari))
  }
  if (!is.null(x$metrics$silhouette))
    cat(sprintf("  Silhouette:  %.4f\n", x$metrics$silhouette))

  # Classification
  if (!is.null(x$metrics$accuracy_mean)) {
    cat("\nClassification (stratified ", x$params$n_folds, "-fold CV):\n", sep = "")
    cat(sprintf("  Mean Accuracy: %.4f\n", x$metrics$accuracy_mean))
    cat(sprintf("  Mean F1:       %.4f\n", x$metrics$f1_mean))
    cat(sprintf("  Mean AUROC:    %.4f\n", x$metrics$auroc_mean))
    for (clf in x$params$classifiers) {
      acc_key <- paste0("accuracy_", clf)
      f1_key <- paste0("f1_", clf)
      if (!is.null(x$metrics[[acc_key]]))
        cat(sprintf("    %-4s  Acc=%.4f  F1=%.4f\n", clf,
                    x$metrics[[acc_key]], x$metrics[[f1_key]]))
    }
  }

  # Batch mixing
  if (!is.null(x$metrics$batch_silhouette)) {
    cat("\nBatch Mixing:\n")
    cat(sprintf("  Batch Silhouette: %+.4f  (lower = better mixing)\n",
                x$metrics$batch_silhouette))
    cat(sprintf("  Batch kNN Entropy: %.4f  (higher = better mixing)\n",
                x$metrics$batch_knn_entropy))
  }

  invisible(x)
}


#' Convert assessment results to a one-row data frame
#' @param x An \code{nmf_assessment} object.
#' @param ... Ignored.
#' @return A one-row data frame with all metrics as columns.
#' @export
as.data.frame.nmf_assessment <- function(x, ...) {
  as.data.frame(x$metrics, stringsAsFactors = FALSE)
}


# ============================================================
# Embedding extraction
# ============================================================

.extract_embedding <- function(x) {
  if (is.matrix(x) || is.data.frame(x))
    return(as.matrix(x))
  if (methods::is(x, "nmf")) {
    # H is k x n (factors x samples), so t(diag(d) %*% H) gives samples x factors
    return(t(x@d * x@h))  # efficient: d * h scales each row of h by d
  }
  if (methods::is(x, "svd")) {
    # u is m x k (samples x factors), d scales singular values
    return(x@u %*% diag(x@d))
  }
  if (inherits(x, "factor_net_result")) {
    layer <- x$layers[[1]]
    return(t(layer$d * layer$H))
  }
  stop("x must be a matrix, nmf, svd, or factor_net_result object")
}


# ============================================================
# Adjusted Rand Index
# ============================================================

.adjusted_rand_index <- function(labels_true, labels_pred) {
  # Contingency table approach
  ct <- table(labels_true, labels_pred)
  sum_comb_i <- sum(choose(rowSums(ct), 2))
  sum_comb_j <- sum(choose(colSums(ct), 2))
  sum_comb_ij <- sum(choose(ct, 2))
  n <- length(labels_true)
  sum_comb_n <- choose(n, 2)

  expected <- sum_comb_i * sum_comb_j / sum_comb_n
  max_index <- 0.5 * (sum_comb_i + sum_comb_j)
  denom <- max_index - expected

  if (denom == 0) return(0)
  (sum_comb_ij - expected) / denom
}


# ============================================================
# Normalized Mutual Information
# ============================================================

.normalized_mutual_info <- function(labels_true, labels_pred) {
  n <- length(labels_true)
  ct <- table(labels_true, labels_pred)

  # Marginal probabilities
  p_i <- rowSums(ct) / n
  p_j <- colSums(ct) / n
  p_ij <- ct / n

  # Entropy
  H_true <- -sum(p_i[p_i > 0] * log(p_i[p_i > 0]))
  H_pred <- -sum(p_j[p_j > 0] * log(p_j[p_j > 0]))

  # Mutual information
  MI <- 0
  for (i in seq_len(nrow(ct))) {
    for (j in seq_len(ncol(ct))) {
      if (p_ij[i, j] > 0 && p_i[i] > 0 && p_j[j] > 0)
        MI <- MI + p_ij[i, j] * log(p_ij[i, j] / (p_i[i] * p_j[j]))
    }
  }

  denom <- sqrt(H_true * H_pred)
  if (denom == 0) return(0)
  MI / denom
}


# ============================================================
# Silhouette Score (approximate, sampled references)
# O(n * spc * n_classes) instead of O(n^2)
# ============================================================

.approx_silhouette <- function(embedding, labels_int, spc = 200L, seed = 42L) {
  n <- nrow(embedding)
  dim_k <- ncol(embedding)
  classes <- sort(unique(labels_int))
  n_classes <- length(classes)
  if (n_classes < 2L) return(0)

  set.seed(seed)

  # Sample spc points per class as reference points
  class_refs <- list()
  for (c in classes) {
    idx <- which(labels_int == c)
    s <- min(spc, length(idx))
    sampled <- idx[sample.int(length(idx), s)]
    class_refs[[as.character(c)]] <- embedding[sampled, , drop = FALSE]
  }

  # For each point, compute a_i (mean dist to own-class samples)
  # and b_i (min over other classes of mean dist to that class's samples)
  sil <- numeric(n)
  for (i in seq_len(n)) {
    my_class <- labels_int[i]
    xi <- embedding[i, ]
    a_i <- 0
    b_i <- Inf

    for (c in classes) {
      refs <- class_refs[[as.character(c)]]
      # Vectorized distance: sqrt(sum((xi - each_ref)^2))
      diffs <- sweep(refs, 2, xi, "-")
      dists <- sqrt(pmax(rowSums(diffs^2), 0))
      mean_d <- mean(dists)

      if (c == my_class) {
        a_i <- mean_d
      } else {
        b_i <- min(b_i, mean_d)
      }
    }

    denom <- max(a_i, b_i)
    sil[i] <- if (denom > 0) (b_i - a_i) / denom else 0
  }

  mean(sil)
}


# ============================================================
# Batch kNN Entropy (approximate, reference-sample based)
# Instead of O(n^2) full distance matrix, samples a reference set
# and finds kNN within that reference. O(n * n_ref) where n_ref << n.
# ============================================================

.approx_batch_knn_entropy <- function(embedding, batch_int, n_batch,
                                       k = 50L, seed = 42L) {
  n <- nrow(embedding)
  k <- min(k, n - 1L)
  dim_k <- ncol(embedding)

  set.seed(seed)

  # Use a random reference subset for neighbor search
  # n_ref = min(n, max(5000, 10*k)) — enough for good kNN approximation
  n_ref <- min(n, max(5000L, 10L * k))
  ref_idx <- sample.int(n, n_ref)
  ref_emb <- embedding[ref_idx, , drop = FALSE]
  ref_batch <- batch_int[ref_idx]

  max_entropy <- log(n_batch)
  entropies <- numeric(n)

  # For each point, find k nearest in reference set
  # This is O(n * n_ref * dim) — manageable for n_ref ~ 5000
  ref_sq_norms <- rowSums(ref_emb^2)

  for (i in seq_len(n)) {
    xi <- embedding[i, ]
    # Squared distances to all reference points
    d2 <- ref_sq_norms - 2 * as.numeric(ref_emb %*% xi) + sum(xi^2)
    # Exclude self if present in reference set
    self_match <- which(ref_idx == i)
    if (length(self_match) > 0L) d2[self_match] <- Inf

    actual_k <- min(k, sum(is.finite(d2)))
    nn_idx <- order(d2)[seq_len(actual_k)]
    nn_batch <- ref_batch[nn_idx]

    counts <- tabulate(nn_batch + 1L, nbins = n_batch)
    props <- counts / sum(counts)
    props <- props[props > 0]
    ent <- -sum(props * log(props))
    entropies[i] <- if (max_entropy > 0) ent / max_entropy else 0
  }

  mean(entropies)
}


# ============================================================
# Stratified K-Fold Cross-Validation Classification
# ============================================================

.cv_classify <- function(embedding, labels, n_folds, classifiers, k_nn, seed) {
  n <- nrow(embedding)
  labels_fac <- as.factor(labels)
  labels_int <- as.integer(labels_fac) - 1L
  classes <- 0:(nlevels(labels_fac) - 1L)
  n_classes <- length(classes)

  set.seed(seed)

  # Stratified fold assignment
  fold_ids <- integer(n)
  for (lev in levels(labels_fac)) {
    idx <- which(labels_fac == lev)
    fold_ids[idx] <- rep_len(sample(n_folds), length(idx))
  }

  results <- list()

  for (fold in seq_len(n_folds)) {
    test_mask <- fold_ids == fold
    train_mask <- !test_mask

    if (sum(test_mask) == 0L || sum(train_mask) == 0L) next

    X_train <- embedding[train_mask, , drop = FALSE]
    X_test <- embedding[test_mask, , drop = FALSE]
    y_train <- labels_int[train_mask]
    y_test <- labels_int[test_mask]
    y_train_fac <- labels_fac[train_mask]

    # Scale features
    means <- colMeans(X_train)
    sds <- apply(X_train, 2, stats::sd)
    sds[sds == 0] <- 1
    X_train_s <- scale(X_train, center = means, scale = sds)
    X_test_s <- scale(X_test, center = means, scale = sds)

    for (clf_name in classifiers) {
      fold_result <- tryCatch({
        switch(clf_name,
          knn = .knn_predict(X_train_s, y_train, X_test_s, y_test, k_nn, n_classes),
          lr  = .lr_predict(X_train_s, y_train, y_train_fac, X_test_s, y_test, n_classes),
          rf  = .rf_predict(X_train, y_train, y_train_fac, X_test, y_test, n_classes)
        )
      }, error = function(e) {
        list(accuracy = NA, f1 = NA, precision = NA, recall = NA, auroc = NA)
      })

      results[[length(results) + 1L]] <- data.frame(
        classifier = clf_name,
        fold = fold,
        accuracy = fold_result$accuracy,
        f1 = fold_result$f1,
        precision = fold_result$precision,
        recall = fold_result$recall,
        auroc = fold_result$auroc,
        stringsAsFactors = FALSE
      )
    }
  }

  list(details = do.call(rbind, results))
}


# --- KNN predictor (vectorized, avoids full n^2 distance matrix) ---
.knn_predict <- function(X_train, y_train, X_test, y_test, k, n_classes) {
  n_test <- nrow(X_test)
  n_train <- nrow(X_train)
  k <- min(k, n_train)
  predictions <- integer(n_test)
  vote_matrix <- matrix(0, n_test, n_classes)

  # Precompute train squared norms for fast distance computation
  train_sq_norms <- rowSums(X_train^2)

  for (i in seq_len(n_test)) {
    xi <- X_test[i, ]
    # d2 = ||xi||^2 + ||xj||^2 - 2*xi'xj (vectorized over all training points)
    d2 <- train_sq_norms - 2 * as.numeric(X_train %*% xi) + sum(xi^2)
    nn_idx <- order(d2)[seq_len(k)]
    nn_labels <- y_train[nn_idx]
    votes <- tabulate(nn_labels + 1L, nbins = n_classes)
    vote_matrix[i, ] <- votes / k
    predictions[i] <- which.max(votes) - 1L
  }

  .compute_clf_metrics(predictions, y_test, vote_matrix, n_classes)
}


# --- Logistic Regression predictor ---
.lr_predict <- function(X_train, y_train, y_train_fac, X_test, y_test, n_classes) {
  # One-vs-rest logistic regression
  n_test <- nrow(X_test)
  prob_matrix <- matrix(0, n_test, n_classes)

  for (c_idx in seq_len(n_classes)) {
    y_bin <- as.integer(y_train == (c_idx - 1L))
    df_train <- data.frame(y = y_bin, as.data.frame(X_train))
    df_test <- as.data.frame(X_test)
    colnames(df_test) <- colnames(df_train)[-1]

    fit <- suppressWarnings(
      stats::glm(y ~ ., data = df_train, family = stats::binomial(link = "logit"))
    )
    prob_matrix[, c_idx] <- suppressWarnings(
      stats::predict(fit, newdata = df_test, type = "response")
    )
  }

  row_sums <- rowSums(prob_matrix)
  row_sums[row_sums == 0] <- 1
  prob_matrix <- prob_matrix / row_sums
  predictions <- apply(prob_matrix, 1, which.max) - 1L

  .compute_clf_metrics(predictions, y_test, prob_matrix, n_classes)
}


# --- Random Forest predictor ---
.rf_predict <- function(X_train, y_train, y_train_fac, X_test, y_test, n_classes) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    return(list(accuracy = NA, f1 = NA, precision = NA, recall = NA, auroc = NA))
  }

  rf <- randomForest::randomForest(x = X_train, y = y_train_fac, ntree = 500L)
  pred_fac <- stats::predict(rf, newdata = X_test, type = "response")
  predictions <- as.integer(pred_fac) - 1L
  prob_matrix <- stats::predict(rf, newdata = X_test, type = "vote")
  if (is.null(dim(prob_matrix)))
    prob_matrix <- matrix(prob_matrix, nrow = nrow(X_test))

  .compute_clf_metrics(predictions, y_test, prob_matrix, n_classes)
}


# ============================================================
# Shared metric computation
# ============================================================

.compute_clf_metrics <- function(predictions, true_labels, prob_matrix, n_classes) {
  n_test <- length(true_labels)
  accuracy <- sum(predictions == true_labels) / n_test

  # Per-class precision, recall, F1
  prec_vals <- numeric(n_classes)
  rec_vals <- numeric(n_classes)
  f1_vals <- numeric(n_classes)

  for (c in seq_len(n_classes)) {
    c_idx <- c - 1L
    tp <- sum(predictions == c_idx & true_labels == c_idx)
    fp <- sum(predictions == c_idx & true_labels != c_idx)
    fn <- sum(predictions != c_idx & true_labels == c_idx)

    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    rec <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1 <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0

    prec_vals[c] <- prec
    rec_vals[c] <- rec
    f1_vals[c] <- f1
  }

  # Macro-averaged AUC (one-vs-rest)
  auc_vals <- numeric(n_classes)
  for (c in seq_len(n_classes)) {
    c_idx <- c - 1L
    y_bin <- as.integer(true_labels == c_idx)
    scores <- if (ncol(prob_matrix) >= c) prob_matrix[, c] else rep(0, n_test)
    auc_vals[c] <- .compute_auc(y_bin, scores)
  }

  list(
    accuracy = accuracy,
    f1 = mean(f1_vals),
    precision = mean(prec_vals),
    recall = mean(rec_vals),
    auroc = mean(auc_vals)
  )
}


# ============================================================
# GPU dispatch helpers
# ============================================================

.assess_gpu_available <- function() {
  tryCatch({
    gpu_available() && .gpu_env$lib_loaded
  }, error = function(e) FALSE)
}

.assess_gpu <- function(embedding, labels_int, n_classes, batch_int, n_batch,
                         do_metrics, n_folds, k_nn, sil_samples_per_class,
                         batch_knn_k, seed) {
  tryCatch({
    n <- nrow(embedding)
    dim_k <- ncol(embedding)

    # Flatten embedding row-major for .C()
    emb_flat <- as.double(as.vector(t(embedding)))

    # Metric flags
    do_clust <- as.integer("ari" %in% do_metrics || "nmi" %in% do_metrics)
    do_sil <- as.integer("silhouette" %in% do_metrics)
    do_clf <- as.integer("classification" %in% do_metrics)
    do_batch <- as.integer("batch_mixing" %in% do_metrics && n_batch > 1L)

    # Output slots
    out_ari <- as.double(0)
    out_nmi <- as.double(0)
    out_sil <- as.double(0)
    out_knn_acc <- as.double(0)
    out_knn_f1 <- as.double(0)
    out_batch_sil <- as.double(0)
    out_batch_ent <- as.double(0)
    out_status <- as.integer(-1L)

    res <- .gpu_call("rcppml_gpu_assess",
      emb_flat,
      as.integer(n), as.integer(dim_k),
      as.integer(labels_int), as.integer(n_classes),
      as.integer(batch_int), as.integer(n_batch),
      do_clust, do_sil, do_clf, do_batch,
      as.integer(10L),   # kmeans_nstart
      as.integer(100L),  # kmeans_maxiter
      as.integer(sil_samples_per_class),
      as.integer(k_nn),
      as.integer(n_folds),
      as.integer(batch_knn_k),
      as.integer(seed),
      out_ari, out_nmi, out_sil,
      out_knn_acc, out_knn_f1,
      out_batch_sil, out_batch_ent,
      out_status
    )

    if (res[[length(res)]] != 0L) return(NULL)

    list(
      ari = res[[19]],
      nmi = res[[20]],
      silhouette = res[[21]],
      knn_accuracy = res[[22]],
      knn_f1 = res[[23]],
      batch_sil = res[[24]],
      batch_entropy = res[[25]]
    )
  }, error = function(e) {
    message("GPU assess failed: ", conditionMessage(e), ". Falling back to CPU.")
    NULL
  })
}
