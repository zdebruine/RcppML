# classifier_metrics.R
#
# Classification evaluation utilities for factor network embeddings.
# Provides KNN classifier with proper train/test evaluation and a full
# suite of metrics: accuracy, precision, recall, F1, AUC (one-vs-rest),
# and confusion matrix.

#' Evaluate classification performance of factor embeddings
#'
#' Trains a k-nearest-neighbor classifier on factor embeddings and evaluates
#' on held-out test samples. Returns a comprehensive metrics object.
#'
#' @param embedding Numeric matrix where rows are samples and columns are
#'   features (e.g., \code{t(result$H)} for sample embeddings).
#' @param labels Integer or factor vector of class labels. Length must equal
#'   \code{nrow(embedding)}.
#' @param test_fraction Fraction of samples held out for testing (default 0.2).
#' @param test_idx Optional integer vector of test indices. If provided,
#'   \code{test_fraction} is ignored.
#' @param k Number of nearest neighbors (default 5).
#' @param seed Random seed for train/test split reproducibility.
#' @param distance Distance metric: "euclidean" (default) or "cosine".
#' @return An \code{fn_classifier_eval} object with fields:
#'   \describe{
#'     \item{accuracy}{Overall test accuracy}
#'     \item{per_class}{Data frame with per-class precision, recall, F1, support}
#'     \item{macro_precision}{Macro-averaged precision}
#'     \item{macro_recall}{Macro-averaged recall}
#'     \item{macro_f1}{Macro-averaged F1}
#'     \item{weighted_f1}{Support-weighted F1}
#'     \item{auc}{Macro-averaged one-vs-rest AUC (from neighbor vote fractions)}
#'     \item{confusion}{Confusion matrix (rows = true, cols = predicted)}
#'     \item{predictions}{Test set predictions}
#'     \item{test_labels}{True test labels}
#'     \item{train_idx}{Training sample indices}
#'     \item{test_idx}{Test sample indices}
#'     \item{k}{Number of neighbors used}
#'   }
#' @examples
#' \donttest{
#' data(digits)
#' model <- nmf(digits, 10, maxit = 20, seed = 1, verbose = FALSE)
#' labels <- attr(digits, "target")
#' eval <- classify_embedding(model$w, labels, test_fraction = 0.2, k = 5, seed = 42)
#' print(eval)
#' }
#' @seealso \code{\link{classify_logistic}}, \code{\link{classify_rf}}
#' @export
classify_embedding <- function(embedding, labels,
                               test_fraction = 0.2,
                               test_idx = NULL,
                               k = 5L,
                               seed = NULL,
                               distance = c("euclidean", "cosine")) {
  distance <- match.arg(distance)
  embedding <- as.matrix(embedding)
  n <- nrow(embedding)

  if (length(labels) != n)
    stop("length(labels) must equal nrow(embedding)")

  # Convert factors to integer labels
  if (is.factor(labels)) {
    label_levels <- levels(labels)
    labels <- as.integer(labels) - 1L
  } else {
    labels <- as.integer(labels)
    label_levels <- as.character(sort(unique(labels)))
  }

  classes <- sort(unique(labels))
  n_classes <- length(classes)
  k <- as.integer(min(k, n - 1L))

  # Train/test split

  if (is.null(test_idx)) {
    if (!is.null(seed)) set.seed(seed)
    n_test <- max(1L, as.integer(round(n * test_fraction)))
    test_idx <- sample.int(n, n_test)
  }
  train_idx <- setdiff(seq_len(n), test_idx)

  if (length(train_idx) < k)
    stop("Fewer training samples than k neighbors")

  train_embed <- embedding[train_idx, , drop = FALSE]
  test_embed <- embedding[test_idx, , drop = FALSE]
  train_labels <- labels[train_idx]
  test_labels <- labels[test_idx]
  n_test <- length(test_idx)

  # Normalize for cosine distance
  if (distance == "cosine") {
    row_norms <- sqrt(rowSums(train_embed^2))
    row_norms[row_norms == 0] <- 1
    train_embed <- train_embed / row_norms
    row_norms_t <- sqrt(rowSums(test_embed^2))
    row_norms_t[row_norms_t == 0] <- 1
    test_embed <- test_embed / row_norms_t
  }

  # KNN classification with vote fractions (for AUC)
  predictions <- integer(n_test)
  vote_matrix <- matrix(0, n_test, n_classes)  # fraction of votes per class
  colnames(vote_matrix) <- as.character(classes)

  for (i in seq_len(n_test)) {
    if (distance == "euclidean") {
      dists <- colSums((t(train_embed) - test_embed[i, ])^2)
    } else {
      # cosine distance = 1 - cosine similarity
      dists <- 1 - (train_embed %*% test_embed[i, ])
    }
    nn_idx <- order(dists)[seq_len(k)]
    nn_labels <- train_labels[nn_idx]

    # Vote counts
    votes <- tabulate(match(nn_labels, classes), nbins = n_classes)
    vote_matrix[i, ] <- votes / k
    predictions[i] <- classes[which.max(votes)]
  }

  result <- .build_eval_result(
    predictions = predictions,
    test_labels = test_labels,
    prob_matrix = vote_matrix,
    classes = classes,
    label_levels = label_levels,
    n_classes = n_classes,
    train_idx = train_idx,
    test_idx = test_idx,
    method = "knn"
  )
  result$k <- k
  result$distance <- distance
  result
}

#' Print a classifier evaluation result
#'
#' Displays a human-readable summary of classification metrics including
#' accuracy, macro/weighted F1, AUC, and per-class precision/recall.
#'
#' @param x An \code{fn_classifier_eval} object returned by
#'   \code{\link{classify_embedding}}, \code{\link{classify_logistic}},
#'   or \code{\link{classify_rf}}.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{summary.fn_classifier_eval}}, \code{\link{classify_embedding}}
#' @method print fn_classifier_eval
#' @export
print.fn_classifier_eval <- function(x, ...) {
  method_str <- if (!is.null(x$k)) {
    sprintf("%d-NN, %s distance", x$k, x$distance)
  } else if (!is.null(x$method)) {
    x$method
  } else {
    "unknown"
  }
  cat(sprintf("Classification Evaluation (%s)\n", method_str))
  cat(sprintf("  Samples: %d train, %d test, %d classes\n",
              length(x$train_idx), length(x$test_idx), x$n_classes))
  cat(sprintf("  Accuracy:  %.4f\n", x$accuracy))
  cat(sprintf("  Macro F1:  %.4f  (P=%.4f, R=%.4f)\n",
              x$macro_f1, x$macro_precision, x$macro_recall))
  cat(sprintf("  Weighted F1: %.4f\n", x$weighted_f1))
  cat(sprintf("  AUC (macro): %.4f\n", x$auc))
  cat("\nPer-class:\n")
  pc <- x$per_class
  pc$precision <- round(pc$precision, 4)
  pc$recall <- round(pc$recall, 4)
  pc$f1 <- round(pc$f1, 4)
  print(pc, row.names = FALSE)
  invisible(x)
}

#' Summarize a classifier evaluation result
#'
#' Returns a tidy data frame of aggregate classification metrics.
#'
#' @param object An \code{fn_classifier_eval} object returned by
#'   \code{\link{classify_embedding}}, \code{\link{classify_logistic}},
#'   or \code{\link{classify_rf}}.
#' @param ... Additional arguments (unused).
#' @return A data frame with columns \code{metric} and \code{value}.
#' @seealso \code{\link{print.fn_classifier_eval}}, \code{\link{classify_embedding}}
#' @method summary fn_classifier_eval
#' @export
summary.fn_classifier_eval <- function(object, ...) {
  data.frame(
    metric = c("accuracy", "macro_precision", "macro_recall",
               "macro_f1", "weighted_f1", "auc"),
    value = c(object$accuracy, object$macro_precision, object$macro_recall,
              object$macro_f1, object$weighted_f1, object$auc)
  )
}


#' Logistic regression classifier for factor embeddings
#'
#' Trains a multinomial logistic regression on factor embeddings using
#' \code{stats::glm} (one-vs-rest for > 2 classes) and evaluates on test
#' samples with the same comprehensive metrics as \code{classify_embedding}.
#'
#' @inheritParams classify_embedding
#' @return An \code{fn_classifier_eval} object (same structure as KNN variant).
#' @examples
#' \donttest{
#' # Logistic regression on random embeddings
#' set.seed(42)
#' embed <- matrix(rnorm(200), nrow = 40, ncol = 5)
#' labels <- factor(rep(1:4, each = 10))
#' eval <- classify_logistic(embed, labels, seed = 1)
#' eval$accuracy
#' }
#' @seealso \code{\link{classify_embedding}}, \code{\link{classify_rf}}
#' @export
classify_logistic <- function(embedding, labels,
                              test_fraction = 0.2,
                              test_idx = NULL,
                              seed = NULL) {
  embedding <- as.matrix(embedding)
  n <- nrow(embedding)
  p <- ncol(embedding)

  if (length(labels) != n)
    stop("length(labels) must equal nrow(embedding)")

  if (is.factor(labels)) {
    label_levels <- levels(labels)
    labels <- as.integer(labels) - 1L
  } else {
    labels <- as.integer(labels)
    label_levels <- as.character(sort(unique(labels)))
  }

  classes <- sort(unique(labels))
  n_classes <- length(classes)

  # Train/test split
  if (is.null(test_idx)) {
    if (!is.null(seed)) set.seed(seed)
    n_test <- max(1L, as.integer(round(n * test_fraction)))
    test_idx <- sample.int(n, n_test)
  }
  train_idx <- setdiff(seq_len(n), test_idx)

  train_embed <- embedding[train_idx, , drop = FALSE]
  test_embed <- embedding[test_idx, , drop = FALSE]
  train_labels <- labels[train_idx]
  test_labels <- labels[test_idx]
  n_test <- length(test_idx)

  # Fit one-vs-rest logistic models
  prob_matrix <- matrix(0, n_test, n_classes)
  colnames(prob_matrix) <- as.character(classes)

  for (c_idx in seq_len(n_classes)) {
    y_bin <- as.integer(train_labels == classes[c_idx])
    df_train <- data.frame(y = y_bin, train_embed)
    df_test <- data.frame(test_embed)
    colnames(df_test) <- colnames(df_train)[-1]

    fit <- suppressWarnings(
      stats::glm(y ~ ., data = df_train, family = stats::binomial(link = "logit"))
    )
    prob_matrix[, c_idx] <- suppressWarnings(
      stats::predict(fit, newdata = df_test, type = "response")
    )
  }

  # Normalize rows to sum to 1 (softmax-like)
  row_sums <- rowSums(prob_matrix)
  row_sums[row_sums == 0] <- 1
  prob_matrix <- prob_matrix / row_sums

  predictions <- classes[apply(prob_matrix, 1, which.max)]

  .build_eval_result(
    predictions = predictions,
    test_labels = test_labels,
    prob_matrix = prob_matrix,
    classes = classes,
    label_levels = label_levels,
    n_classes = n_classes,
    train_idx = train_idx,
    test_idx = test_idx,
    method = "logistic"
  )
}


#' Random forest classifier for factor embeddings
#'
#' Trains a random forest on factor embeddings using the \pkg{randomForest}
#' package (must be installed) and evaluates on test samples.
#'
#' @inheritParams classify_embedding
#' @param ntree Number of trees (default 500).
#' @return An \code{fn_classifier_eval} object (same structure as KNN variant).
#' @examples
#' \donttest{
#' # Random forest on random embeddings (requires randomForest package)
#' if (requireNamespace("randomForest", quietly = TRUE)) {
#'   set.seed(42)
#'   embed <- matrix(rnorm(200), nrow = 40, ncol = 5)
#'   labels <- factor(rep(1:4, each = 10))
#'   eval <- classify_rf(embed, labels, ntree = 100, seed = 1)
#'   eval$accuracy
#' }
#' }
#' @seealso \code{\link{classify_embedding}}, \code{\link{classify_logistic}}
#' @export
classify_rf <- function(embedding, labels,
                        test_fraction = 0.2,
                        test_idx = NULL,
                        ntree = 500L,
                        seed = NULL) {
  if (!requireNamespace("randomForest", quietly = TRUE))
    stop("Package 'randomForest' is required for classify_rf(). ",
         "Install with: install.packages('randomForest')")

  embedding <- as.matrix(embedding)
  n <- nrow(embedding)

  if (length(labels) != n)
    stop("length(labels) must equal nrow(embedding)")

  if (is.factor(labels)) {
    label_levels <- levels(labels)
  } else {
    label_levels <- as.character(sort(unique(labels)))
    labels <- factor(labels, levels = label_levels)
  }

  classes <- sort(unique(as.integer(labels) - 1L))
  n_classes <- length(classes)

  # Train/test split
  if (is.null(test_idx)) {
    if (!is.null(seed)) set.seed(seed)
    n_test <- max(1L, as.integer(round(n * test_fraction)))
    test_idx <- sample.int(n, n_test)
  }
  train_idx <- setdiff(seq_len(n), test_idx)

  train_embed <- embedding[train_idx, , drop = FALSE]
  test_embed <- embedding[test_idx, , drop = FALSE]
  train_labels <- labels[train_idx]
  test_labels_fac <- labels[test_idx]
  test_labels <- as.integer(test_labels_fac) - 1L
  n_test <- length(test_idx)

  rf <- randomForest::randomForest(
    x = train_embed, y = train_labels,
    ntree = as.integer(ntree)
  )

  pred_fac <- stats::predict(rf, newdata = test_embed, type = "response")
  predictions <- as.integer(pred_fac) - 1L

  # Class probabilities from vote fractions
  prob_matrix <- stats::predict(rf, newdata = test_embed, type = "vote")
  if (is.null(dim(prob_matrix))) {
    prob_matrix <- matrix(prob_matrix, nrow = n_test)
  }

  .build_eval_result(
    predictions = predictions,
    test_labels = test_labels,
    prob_matrix = prob_matrix,
    classes = classes,
    label_levels = label_levels,
    n_classes = n_classes,
    train_idx = train_idx,
    test_idx = test_idx,
    method = "random_forest"
  )
}


# ============================================================================
# Internal: build fn_classifier_eval from predictions + probabilities
# ============================================================================

.build_eval_result <- function(predictions, test_labels, prob_matrix,
                               classes, label_levels, n_classes,
                               train_idx, test_idx, method) {
  n_test <- length(test_labels)

  # Confusion matrix
  confusion <- matrix(0L, n_classes, n_classes,
                      dimnames = list(true = label_levels, predicted = label_levels))
  for (i in seq_len(n_test)) {
    r <- match(test_labels[i], classes)
    c <- match(predictions[i], classes)
    if (!is.na(r) && !is.na(c))
      confusion[r, c] <- confusion[r, c] + 1L
  }

  # Per-class metrics
  per_class <- data.frame(
    class = label_levels,
    precision = numeric(n_classes),
    recall = numeric(n_classes),
    f1 = numeric(n_classes),
    support = integer(n_classes),
    stringsAsFactors = FALSE
  )

  for (c_idx in seq_len(n_classes)) {
    tp <- confusion[c_idx, c_idx]
    fp <- sum(confusion[, c_idx]) - tp
    fn <- sum(confusion[c_idx, ]) - tp
    support <- sum(confusion[c_idx, ])

    prec <- if ((tp + fp) > 0) tp / (tp + fp) else 0
    rec <- if ((tp + fn) > 0) tp / (tp + fn) else 0
    f1 <- if ((prec + rec) > 0) 2 * prec * rec / (prec + rec) else 0

    per_class$precision[c_idx] <- prec
    per_class$recall[c_idx] <- rec
    per_class$f1[c_idx] <- f1
    per_class$support[c_idx] <- support
  }

  macro_prec <- mean(per_class$precision)
  macro_rec <- mean(per_class$recall)
  macro_f1 <- mean(per_class$f1)
  total_support <- sum(per_class$support)
  weighted_f1 <- sum(per_class$f1 * per_class$support) / max(total_support, 1)

  # One-vs-rest AUC from probability matrix
  auc_vals <- numeric(n_classes)
  for (c_idx in seq_len(n_classes)) {
    y_true <- as.integer(test_labels == classes[c_idx])
    scores <- if (ncol(prob_matrix) >= c_idx) prob_matrix[, c_idx] else rep(0, n_test)
    auc_vals[c_idx] <- .compute_auc(y_true, scores)
  }
  macro_auc <- mean(auc_vals)

  accuracy <- sum(predictions == test_labels) / n_test

  structure(list(
    accuracy = accuracy,
    per_class = per_class,
    macro_precision = macro_prec,
    macro_recall = macro_rec,
    macro_f1 = macro_f1,
    weighted_f1 = weighted_f1,
    auc = macro_auc,
    per_class_auc = setNames(auc_vals, label_levels),
    confusion = confusion,
    predictions = predictions,
    test_labels = test_labels,
    train_idx = train_idx,
    test_idx = test_idx,
    method = method,
    n_classes = n_classes
  ), class = "fn_classifier_eval")
}


# ============================================================================
# Internal: trapezoidal-rule AUC for binary classification
# ============================================================================

.compute_auc <- function(y_true, scores) {
  n <- length(y_true)
  if (n == 0 || length(unique(y_true)) < 2) return(0.5)

  # Sort by decreasing score
  ord <- order(scores, decreasing = TRUE)
  y_sorted <- y_true[ord]

  n_pos <- sum(y_true == 1)
  n_neg <- n - n_pos
  if (n_pos == 0 || n_neg == 0) return(0.5)

  # Wilcoxon-Mann-Whitney statistic
  tp <- 0
  auc <- 0
  for (i in seq_len(n)) {
    if (y_sorted[i] == 1) {
      tp <- tp + 1
    } else {
      auc <- auc + tp
    }
  }
  auc / (n_pos * n_neg)
}
