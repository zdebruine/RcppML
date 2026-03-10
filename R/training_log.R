# training_log.R
#
# Structured per-iteration training logger for factor_net fitting.
# Records loss, factor norms, regularization penalties, classifier
# performance, and wall time — analogous to TensorBoard logging.

#' Create a training logger for factor network fitting
#'
#' Returns a logger object that records per-iteration metrics during
#' \code{fit()}. After training, the log can be printed, plotted, or
#' exported to CSV.
#'
#' @param log_loss Log reconstruction loss per layer (default TRUE).
#' @param log_norms Log factor W/H norms per layer (default FALSE).
#' @param log_classifier Evaluate classifier accuracy per iteration
#'   using a supplied \code{classify_embedding} configuration.
#'   Must be a list with \code{labels} and optionally \code{test_idx},
#'   \code{k}, \code{side}, and \code{layer} (default NULL = disabled).
#' @param interval Log every \code{interval} iterations (default 1).
#' @return A \code{training_logger} object to pass to \code{fit()}.
#' @examples
#' \dontrun{
#' logger <- training_logger(
#'   log_norms = TRUE,
#'   log_classifier = list(labels = labels, test_idx = test_idx, k = 5)
#' )
#' res <- fit(net, logger = logger)
#' print(logger)
#' plot(logger)
#' export_log(logger, "training_log.csv")
#' }
#' @seealso \code{\link{nmf}}, \code{\link{fit}}
#' @export
training_logger <- function(log_loss = TRUE,
                            log_norms = FALSE,
                            log_classifier = NULL,
                            interval = 1L) {
  if (!is.null(log_classifier)) {
    if (!is.list(log_classifier))
      stop("log_classifier must be a list with at least 'labels'")
    if (is.null(log_classifier$labels))
      stop("log_classifier must contain 'labels'")
    if (is.null(log_classifier$k)) log_classifier$k <- 5L
    if (is.null(log_classifier$side)) log_classifier$side <- "H"
    if (is.null(log_classifier$layer)) log_classifier$layer <- NULL
    if (is.null(log_classifier$method)) log_classifier$method <- "knn"
  }

  logger <- list(
    log_loss = log_loss,
    log_norms = log_norms,
    log_classifier = log_classifier,
    interval = as.integer(max(1L, interval)),
    entries = list(),
    start_time = NULL
  )
  class(logger) <- "training_logger"
  logger
}

# ============================================================================
# Internal: record a snapshot (called from .fn_fit_deep)
# ============================================================================

.log_snapshot <- function(logger, iter, layer_states, layers,
                          per_layer_loss = NULL) {
  if (is.null(logger)) return(logger)
  if (iter %% logger$interval != 0) return(logger)

  if (is.null(logger$start_time)) {
    logger$start_time <- proc.time()["elapsed"]
  }
  elapsed <- proc.time()["elapsed"] - logger$start_time

  n_layers <- length(layer_states)
  entry <- list(
    iteration = iter,
    wall_sec = elapsed
  )

  # --- Per-layer loss ---
  if (logger$log_loss && !is.null(per_layer_loss)) {
    entry$loss <- per_layer_loss
    entry$total_loss <- sum(per_layer_loss)
  }

  # --- Factor norms ---
  if (logger$log_norms) {
    norms <- list()
    for (i in seq_len(n_layers)) {
      nm <- if (!is.null(layers[[i]]$name)) layers[[i]]$name else paste0("L", i)
      s <- layer_states[[i]]
      norms[[nm]] <- list(
        W_frobenius = sqrt(sum(s$W^2)),
        H_frobenius = sqrt(sum(s$H^2)),
        d_sum = sum(s$d),
        d_max = max(s$d),
        d_min = min(s$d),
        W_sparsity = mean(s$W == 0),
        H_sparsity = mean(s$H == 0)
      )
    }
    entry$norms <- norms
  }

  # --- Classifier evaluation ---
  if (!is.null(logger$log_classifier)) {
    clf <- logger$log_classifier
    target_layer <- clf$layer
    if (is.null(target_layer)) target_layer <- n_layers

    # Resolve layer index
    if (is.character(target_layer)) {
      idx <- which(vapply(seq_len(n_layers), function(i) {
        nm <- layers[[i]]$name
        !is.null(nm) && nm == target_layer
      }, logical(1)))
      if (length(idx) == 0) idx <- n_layers
      target_layer <- idx[1]
    }

    s <- layer_states[[target_layer]]
    if (clf$side == "H") {
      embed <- t(s$H)  # n x k
    } else {
      embed <- s$W     # m x k
    }

    eval_result <- tryCatch({
      if (clf$method == "logistic") {
        classify_logistic(embed, clf$labels,
                          test_idx = clf$test_idx, seed = 42L)
      } else if (clf$method == "rf") {
        classify_rf(embed, clf$labels,
                    test_idx = clf$test_idx, seed = 42L)
      } else {
        classify_embedding(embed, clf$labels,
                           test_idx = clf$test_idx,
                           k = clf$k, seed = 42L)
      }
    }, error = function(e) NULL)

    if (!is.null(eval_result)) {
      entry$classifier <- list(
        accuracy = eval_result$accuracy,
        macro_f1 = eval_result$macro_f1,
        auc = eval_result$auc,
        macro_precision = eval_result$macro_precision,
        macro_recall = eval_result$macro_recall
      )
    }
  }

  logger$entries[[length(logger$entries) + 1L]] <- entry
  logger
}

# ============================================================================
# Export logger to data.frame
# ============================================================================

#' Convert training log to data.frame
#'
#' Converts the internal training log from an NMF run into a tidy
#' data frame with one row per logged iteration, suitable for plotting
#' convergence curves.
#'
#' @param x A \code{training_logger} object (from \code{nmf(..., log = TRUE)}).
#' @param ... Ignored.
#' @return A data.frame with columns \code{iteration}, \code{wall_sec},
#'   and optionally \code{total_loss}, per-layer loss columns, and
#'   norm-tracking columns.
#' @seealso \code{\link{nmf}}, \code{\link{plot.training_logger}}
#' @method as.data.frame training_logger
#' @export
as.data.frame.training_logger <- function(x, ...) {
  if (length(x$entries) == 0)
    return(data.frame(iteration = integer(0), wall_sec = numeric(0)))

  rows <- lapply(x$entries, function(e) {
    row <- data.frame(iteration = e$iteration, wall_sec = e$wall_sec,
                      stringsAsFactors = FALSE)
    if (!is.null(e$total_loss)) row$total_loss <- e$total_loss
    if (!is.null(e$loss)) {
      for (i in seq_along(e$loss)) {
        row[[paste0("loss_L", i)]] <- e$loss[i]
      }
    }
    if (!is.null(e$norms)) {
      for (nm in names(e$norms)) {
        for (metric in names(e$norms[[nm]])) {
          row[[paste0(nm, "_", metric)]] <- e$norms[[nm]][[metric]]
        }
      }
    }
    if (!is.null(e$classifier)) {
      for (metric in names(e$classifier)) {
        row[[paste0("clf_", metric)]] <- e$classifier[[metric]]
      }
    }
    row
  })

  # Align columns across all rows
  all_cols <- unique(unlist(lapply(rows, names)))
  aligned <- lapply(rows, function(r) {
    missing <- setdiff(all_cols, names(r))
    for (col in missing) r[[col]] <- NA
    r[all_cols]
  })
  do.call(rbind, aligned)
}

#' Export training log to CSV
#'
#' @param logger A \code{training_logger} object.
#' @param file Path to write the CSV file.
#' @return Invisibly returns the data frame written.
#' @examples
#' \donttest{
#' logger <- training_logger()
#' # After fitting: export_log(logger, tempfile(fileext = ".csv"))
#' }
#' @seealso \code{\link{training_logger}}, \code{\link{as.data.frame.training_logger}}
#' @export
export_log <- function(logger, file) {
  df <- as.data.frame(logger)
  utils::write.csv(df, file, row.names = FALSE)
  invisible(df)
}

# ============================================================================
# Print and plot
# ============================================================================

#' Print a training log
#' @param x A \code{training_logger} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{training_logger}}
#' @method print training_logger
#' @export
print.training_logger <- function(x, ...) {
  n <- length(x$entries)
  cat(sprintf("training_logger: %d entries\n", n))
  if (n == 0) return(invisible(x))

  last <- x$entries[[n]]
  cat(sprintf("  Iterations: %d\n", last$iteration))
  cat(sprintf("  Wall time: %.1f sec\n", last$wall_sec))
  if (!is.null(last$total_loss))
    cat(sprintf("  Final loss: %g\n", last$total_loss))
  if (!is.null(last$classifier))
    cat(sprintf("  Final accuracy: %.4f  F1: %.4f  AUC: %.4f\n",
                last$classifier$accuracy,
                last$classifier$macro_f1,
                last$classifier$auc))
  if (!is.null(last$norms)) {
    cat("  Factor norms (last iter):\n")
    for (nm in names(last$norms)) {
      nrm <- last$norms[[nm]]
      cat(sprintf("    %s: ||W||=%.3f ||H||=%.3f d=[%.3f,%.3f]\n",
                  nm, nrm$W_frobenius, nrm$H_frobenius,
                  nrm$d_min, nrm$d_max))
    }
  }
  invisible(x)
}

#' Plot training log
#'
#' Produces a multi-panel plot of loss, classifier metrics, and factor norms
#' over training iterations.
#'
#' @param x A \code{training_logger} object.
#' @param ... Ignored.
#' @return Invisibly returns \code{NULL}. Called for its side effect (plotting).
#' @seealso \code{\link{training_logger}}
#' @method plot training_logger
#' @export
plot.training_logger <- function(x, ...) {
  df <- as.data.frame(x)
  if (nrow(df) == 0) {
    message("No log entries to plot")
    return(invisible(NULL))
  }

  # Count panels
  has_loss <- "total_loss" %in% names(df)
  has_clf <- "clf_accuracy" %in% names(df)
  norm_cols <- grep("_frobenius$", names(df), value = TRUE)
  has_norms <- length(norm_cols) > 0
  n_panels <- has_loss + has_clf + has_norms

  if (n_panels == 0) {
    message("No plottable metrics in log")
    return(invisible(NULL))
  }

  old_par <- graphics::par(mfrow = c(n_panels, 1), mar = c(4, 4, 2, 1))
  on.exit(graphics::par(old_par))

  # Loss panel
  if (has_loss) {
    loss_cols <- grep("^loss_L|^total_loss$", names(df), value = TRUE)
    yrange <- range(df[, loss_cols], na.rm = TRUE)
    graphics::plot(df$iteration, df$total_loss, type = "l", lwd = 2,
                   xlab = "Iteration", ylab = "Loss", main = "Training Loss",
                   ylim = yrange)
    per_layer <- grep("^loss_L", loss_cols, value = TRUE)
    if (length(per_layer) > 1) {
      cols <- grDevices::rainbow(length(per_layer))
      for (j in seq_along(per_layer)) {
        graphics::lines(df$iteration, df[[per_layer[j]]], col = cols[j], lty = 2)
      }
      graphics::legend("topright", c("total", per_layer),
                       col = c("black", cols), lty = c(1, rep(2, length(per_layer))),
                       lwd = c(2, rep(1, length(per_layer))), cex = 0.7)
    }
  }

  # Classifier panel
  if (has_clf) {
    clf_cols <- grep("^clf_", names(df), value = TRUE)
    yrange <- c(0, 1)
    graphics::plot(df$iteration, df$clf_accuracy, type = "l", lwd = 2,
                   col = "blue", xlab = "Iteration", ylab = "Score",
                   main = "Classifier Performance", ylim = yrange)
    if ("clf_macro_f1" %in% names(df))
      graphics::lines(df$iteration, df$clf_macro_f1, col = "red", lwd = 2)
    if ("clf_auc" %in% names(df))
      graphics::lines(df$iteration, df$clf_auc, col = "darkgreen", lwd = 2)
    graphics::legend("bottomright",
                     c("Accuracy", "Macro F1", "AUC"),
                     col = c("blue", "red", "darkgreen"), lwd = 2, cex = 0.7)
  }

  # Norms panel
  if (has_norms) {
    yrange <- range(df[, norm_cols], na.rm = TRUE)
    graphics::plot(df$iteration, df[[norm_cols[1]]], type = "l", lwd = 2,
                   xlab = "Iteration", ylab = "Frobenius Norm",
                   main = "Factor Norms", ylim = yrange)
    if (length(norm_cols) > 1) {
      cols <- grDevices::rainbow(length(norm_cols))
      for (j in seq_along(norm_cols)) {
        graphics::lines(df$iteration, df[[norm_cols[j]]], col = cols[j], lwd = 2)
      }
      graphics::legend("topright", norm_cols, col = cols, lwd = 2, cex = 0.7)
    }
  }

  invisible(df)
}
