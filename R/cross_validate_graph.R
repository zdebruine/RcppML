# cross_validate_graph.R
#
# Cross-validation for factor_net graphs. Supports:
#   - Single-rank CV (holdout test loss for one architecture)
#   - Multi-rank CV (compare multiple ranks)
#   - Grid/random search over hyperparameters

# ============================================================================
# cross_validate_graph() -- main entry point
# ============================================================================

#' Cross-validate a factorization network
#'
#' Evaluates a factor_net architecture across a grid of hyperparameter
#' combinations using held-out test loss. Each combination is fitted with
#' a fraction of entries masked, and the test loss on those entries
#' determines the optimal configuration.
#'
#' For single-parameter rank selection, pass \code{k = c(5, 10, 20)}.
#' For multi-parameter search, use \code{params} to specify named lists
#' of values for each layer and parameter.
#'
#' @param inputs Input node(s) from \code{factor_input()}. Can be a
#'   single node or a list of nodes for multi-modal.
#' @param layer_fn A function that, given a named list of parameter values,
#'   returns the output layer node of the network. Example:
#'   \code{function(p) inputs |> nmf_layer(k = p$k, L1 = p$L1)}
#' @param params A named list of parameter vectors to search over.
#'   Names should match the arguments expected by \code{layer_fn}.
#'   Example: \code{list(k = c(5, 10, 20), L1 = c(0, 0.01))}.
#' @param config A \code{fn_global_config} from \code{factor_config()}.
#'   The \code{test_fraction}, \code{cv_seed}, \code{mask_zeros},
#'   and \code{patience} fields are used for cross-validation.
#'   If \code{test_fraction} is 0 (default), it is automatically
#'   set to 0.1.
#' @param reps Number of CV replicates per parameter combination (each
#'   with a different CV mask seed). Default 3.
#' @param strategy Search strategy: \code{"grid"} (all combinations) or
#'   \code{"random"} (sample \code{n_random} combinations). Default "grid".
#' @param n_random Number of combinations for random search. Ignored
#'   for grid search. Default 20.
#' @param seed Seed for random search sampling and CV mask derivation.
#'   Default 42.
#' @param verbose Print progress updates. Default TRUE.
#' @return A \code{factor_net_cv} object with components:
#'   \describe{
#'     \item{results}{Data frame with columns: param values, rep, test_loss,
#'       train_loss, iterations, converged.}
#'     \item{summary}{Data frame with param values, mean_test_loss,
#'       se_test_loss, mean_train_loss, ranked by mean_test_loss.}
#'     \item{best_params}{Named list of the best parameter combination.}
#'     \item{all_fits}{List of all fit results (if \code{keep_fits = TRUE}).}
#'   }
#' @examples
#' \dontrun{
#' library(Matrix)
#' X <- rsparsematrix(100, 50, 0.1)
#' inp <- factor_input(X, "X")
#'
#' # Rank selection
#' cv <- cross_validate_graph(
#'   inputs = inp,
#'   layer_fn = function(p) inp |> nmf_layer(k = p$k),
#'   params = list(k = c(3, 5, 10, 20)),
#'   config = factor_config(maxit = 50, seed = 42)
#' )
#' print(cv)
#' cv$best_params  # optimal rank
#'
#' # Multi-parameter search
#' cv2 <- cross_validate_graph(
#'   inputs = inp,
#'   layer_fn = function(p) inp |> nmf_layer(k = p$k, L1 = p$L1),
#'   params = list(k = c(5, 10, 20), L1 = c(0, 0.01, 0.1)),
#'   config = factor_config(maxit = 50, seed = 42),
#'   reps = 3
#' )
#' }
#' @note The \code{seed} parameter defaults to \code{42L} (deterministic)
#' rather than \code{NULL} (random) used by \code{\link{nmf}()} and
#' \code{\link{svd}()}. This ensures reproducible cross-validation grid
#' searches by default. Pass \code{seed = NULL} for non-deterministic
#' behavior.
#' @seealso \code{\link{factor_net}}, \code{\link{fit}}, \code{\link{factor_config}}
#' @export
cross_validate_graph <- function(inputs, layer_fn, params,
                                  config = factor_config(),
                                  reps = 3L, strategy = c("grid", "random"),
                                  n_random = 20L, seed = 42L,
                                  verbose = TRUE) {
  strategy <- match.arg(strategy)

  if (!is.function(layer_fn))
    stop("'layer_fn' must be a function(p) returning the output layer node")
  if (!is.list(params) || length(params) == 0)
    stop("'params' must be a non-empty named list of parameter vectors")
  if (is.null(names(params)) || any(names(params) == ""))
    stop("All elements of 'params' must be named")

  # Ensure CV is enabled (support both new and old field names)
  tf <- if (!is.null(config$test_fraction)) config$test_fraction
        else if (!is.null(config$holdout_fraction)) config$holdout_fraction
        else 0
  if (tf == 0) tf <- 0.1
  config$test_fraction <- tf

  # Normalize inputs
  if (inherits(inputs, "fn_node")) inputs <- list(inputs)

  # Build parameter grid
  param_grid <- expand.grid(params, stringsAsFactors = FALSE)
  if (strategy == "random" && nrow(param_grid) > n_random) {
    set.seed(seed)
    param_grid <- param_grid[sample.int(nrow(param_grid), n_random), , drop = FALSE]
    rownames(param_grid) <- NULL
  }

  n_combos <- nrow(param_grid)
  if (verbose)
    cat(sprintf("Cross-validating %d parameter combinations x %d reps = %d fits\n",
                n_combos, reps, n_combos * reps))

  # Run all combinations x replicates
  results <- vector("list", n_combos * reps)
  idx <- 0L

  for (ci in seq_len(n_combos)) {
    p <- as.list(param_grid[ci, , drop = FALSE])

    if (verbose)
      cat(sprintf("  [%d/%d] %s\n", ci, n_combos,
                  paste(names(p), "=", p, collapse = ", ")))

    for (ri in seq_len(reps)) {
      idx <- idx + 1L

      # Per-rep CV seed
      rep_cv_seed <- as.integer(seed + (ci - 1L) * reps + ri)
      cv_config <- config
      cv_config$cv_seed <- rep_cv_seed

      # Build the network for this parameter combination
      output <- tryCatch(layer_fn(p), error = function(e) {
        warning(sprintf("layer_fn failed for combo %d: %s", ci, e$message))
        NULL
      })
      if (is.null(output)) {
        results[[idx]] <- data.frame(
          combo = ci, rep = ri,
          test_loss = NA_real_, train_loss = NA_real_,
          iterations = NA_integer_, converged = FALSE,
          stringsAsFactors = FALSE
        )
        next
      }

      net <- factor_net(inputs = inputs, output = output, config = cv_config)
      res <- tryCatch(fit(net), error = function(e) {
        warning(sprintf("fit() failed for combo %d, rep %d: %s",
                        ci, ri, e$message))
        NULL
      })

      if (is.null(res)) {
        results[[idx]] <- data.frame(
          combo = ci, rep = ri,
          test_loss = NA_real_, train_loss = NA_real_,
          iterations = NA_integer_, converged = FALSE,
          stringsAsFactors = FALSE
        )
      } else {
        # Extract test loss from first layer
        first_layer <- res$layers[[1]]
        results[[idx]] <- data.frame(
          combo = ci, rep = ri,
          test_loss = first_layer$test_loss,
          train_loss = first_layer$loss,
          iterations = first_layer$iterations,
          converged = first_layer$converged,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  # Assemble results data frame
  results_df <- do.call(rbind, results)
  # Add parameter columns
  param_cols <- param_grid[results_df$combo, , drop = FALSE]
  rownames(param_cols) <- NULL
  results_df <- cbind(param_cols, results_df)

  # Compute summary (mean +/- SE of test loss per combo)
  summary_list <- lapply(seq_len(n_combos), function(ci) {
    rows <- results_df[results_df$combo == ci, , drop = FALSE]
    tl <- rows$test_loss[!is.na(rows$test_loss)]
    data.frame(
      param_grid[ci, , drop = FALSE],
      mean_test_loss = if (length(tl) > 0) mean(tl) else NA_real_,
      se_test_loss = if (length(tl) > 1) sd(tl) / sqrt(length(tl)) else NA_real_,
      mean_train_loss = mean(rows$train_loss, na.rm = TRUE),
      n_valid = length(tl),
      stringsAsFactors = FALSE
    )
  })
  summary_df <- do.call(rbind, summary_list)
  summary_df <- summary_df[order(summary_df$mean_test_loss), , drop = FALSE]
  rownames(summary_df) <- NULL

  # Best params
  best_idx <- which.min(summary_df$mean_test_loss)
  best_params <- as.list(summary_df[best_idx, names(params), drop = FALSE])

  if (verbose) {
    cat(sprintf("\nBest: %s -> test_loss = %.6f (SE = %.6f)\n",
                paste(names(best_params), "=", best_params, collapse = ", "),
                summary_df$mean_test_loss[best_idx],
                summary_df$se_test_loss[best_idx]))
  }

  structure(list(
    results = results_df,
    summary = summary_df,
    best_params = best_params,
    config = config,
    params = params,
    strategy = strategy,
    reps = reps
  ), class = "factor_net_cv")
}

#' Print a factor_net_cv result
#' @param x A \code{factor_net_cv} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @method print factor_net_cv
#' @export
print.factor_net_cv <- function(x, ...) {
  cat("factor_net cross-validation\n")
  cat(sprintf("  Strategy: %s | Reps: %d | Combos: %d\n",
              x$strategy, x$reps, nrow(x$summary)))
  tf_val <- if (!is.null(x$config$test_fraction)) x$config$test_fraction
            else if (!is.null(x$config$holdout_fraction)) x$config$holdout_fraction
            else 0
  cat(sprintf("  Holdout: %.1f%%\n", tf_val * 100))
  cat("\nRanked results (by mean test loss):\n")
  print(x$summary, row.names = FALSE)
  cat(sprintf("\nBest: %s\n",
              paste(names(x$best_params), "=", x$best_params, collapse = ", ")))
  invisible(x)
}
