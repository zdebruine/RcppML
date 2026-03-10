# factor_methods.R
#
# Execution and accessor methods for factor_net objects.
# fit() delegates to existing nmf() for single-layer networks
# and uses R-level outer ALS for multi-layer networks.

# ============================================================================
# fit() — execute a compiled factor_net
# ============================================================================

#' Fit a factorization network
#'
#' Executes the compiled graph from \code{factor_net()}. For single-layer
#' networks, this delegates directly to \code{nmf()} for maximum performance.
#' Multi-layer networks use R-level alternating least squares.
#'
#' @param object A \code{factor_net} object from \code{factor_net()}.
#' @param logger Optional \code{training_logger} from \code{training_logger()}
#'   for per-iteration diagnostics (loss, norms, classifier metrics).
#' @param ... Additional arguments (currently unused).
#' @return A \code{factor_net_result} object. Access per-layer results by
#'   name (e.g. \code{result$L1}). Each layer is a list with components
#'   \code{W}, \code{d}, \code{H}, and \code{iterations}. Additional
#'   fields: \code{n_layers}, \code{multi_modal}, \code{total_iterations},
#'   \code{total_loss}, and \code{converged}.
#' @seealso \code{\link{factor_net}}, \code{\link{predict.factor_net_result}},
#'   \code{\link{cross_validate_graph}}
#' @examples
#' \dontrun{
#' library(Matrix)
#' X <- rsparsematrix(100, 50, 0.1)
#' inp <- factor_input(X, "X")
#' L1 <- inp |> nmf_layer(k = 5, name = "L1")
#' net <- factor_net(inputs = inp, output = L1,
#'                   config = factor_config(maxit = 100, seed = 42))
#' res <- fit(net)
#' res$L1$W   # m x k basis matrix
#' res$L1$H   # k x n coefficient matrix
#' }
#' @export
fit <- function(object, ...) UseMethod("fit")

#' @rdname fit
#' @export
fit.factor_net <- function(object, logger = NULL, ...) {
  layers <- object$layers
  config <- object$config

  if (length(layers) == 0)
    stop("factor_net has no layers to fit")

  if (!is.null(logger) && !inherits(logger, "training_logger"))
    stop("logger must be a training_logger object from training_logger()")

  # Check if we need R-side callbacks (reference/callback guides)
  needs_r_callbacks <- .fn_has_r_callbacks(layers)

  if (needs_r_callbacks) {
    # Fall back to R-level outer ALS for reference/callback guides
    if (length(layers) == 1)
      return(.fn_fit_single_layer(object, logger = logger))
    return(.fn_fit_deep(object, logger = logger))
  }

  # --- C++ graph path: build descriptor and dispatch ---
  descriptor <- .fn_build_descriptor(object)
  raw <- Rcpp_factor_net_fit(descriptor)
  res <- .fn_wrap_graph_result(raw, object)

  # Logger integration (post-hoc snapshot from C++ results)
  if (!is.null(logger)) {
    layer_states <- lapply(names(res$layers), function(nm) {
      lr <- res$layers[[nm]]
      list(W = lr$W, d = lr$d, H = lr$H)
    })
    per_layer_loss <- vapply(res$layers, function(lr) lr$loss, numeric(1))
    logger <- .log_snapshot(logger, res$total_iterations, layer_states,
                            layers, per_layer_loss)
    res$logger <- logger
  }
  res
}


# ============================================================================
# Single-layer execution: delegate to nmf()
# ============================================================================

.fn_fit_single_layer <- function(net, logger = NULL) {
  layer <- net$layers[[1]]
  config <- net$config
  input_node <- layer$input

  # --- Resolve data matrix ---
  if (input_node$type == "shared") {
    return(.fn_fit_shared_layer(net, logger = logger))
  }

  data <- .fn_resolve_data(input_node)

  # --- Build nmf() arguments from config hierarchy ---
  args <- .fn_build_nmf_args(data, layer, config)

  # --- Call nmf() ---
  result <- do.call(nmf, args)

  # Log final state if logger provided
  if (!is.null(logger)) {
    layer_states <- list(list(W = result@w, d = result@d, H = result@h))
    per_layer_loss <- c(result@misc$loss)
    logger <- .log_snapshot(logger, result@misc$iter, layer_states,
                            list(layer), per_layer_loss)
  }

  # --- Package as factor_net_result ---
  res <- .fn_wrap_single_result(result, layer, net)
  if (!is.null(logger)) res$logger <- logger
  res
}


# ============================================================================
# Multi-modal shared-H: concatenate inputs, call nmf(), split W
# ============================================================================

.fn_fit_shared_layer <- function(net, logger = NULL) {
  layer <- net$layers[[1]]
  config <- net$config
  shared_node <- layer$input

  # Collect input matrices
  input_data <- lapply(shared_node$inputs, .fn_resolve_data)
  input_names <- shared_node$input_names
  input_nrows <- vapply(input_data, nrow, integer(1))

  # Validate all inputs have the same number of columns
  ncols <- vapply(input_data, ncol, integer(1))
  if (length(unique(ncols)) != 1)
    stop("All inputs to factor_shared() must have the same number of columns")

  # Check sparsity consistency
  all_sparse <- all(vapply(input_data, inherits, logical(1), "dgCMatrix"))
  all_dense <- all(vapply(input_data, is.matrix, logical(1)))

  # Concatenate row-wise
  if (all_sparse) {
    data_concat <- do.call(rbind, input_data)
  } else {
    # Convert all to dense for concatenation
    data_concat <- do.call(rbind, lapply(input_data, as.matrix))
  }

  # --- Build nmf() arguments ---
  args <- .fn_build_nmf_args(data_concat, layer, config)

  # --- Call nmf() ---
  result <- do.call(nmf, args)

  # --- Split W back by input dimensions ---
  W_full <- result@w
  d_vec <- result@d
  H_mat <- result@h

  W_list <- list()
  row_start <- 1L
  for (i in seq_along(input_data)) {
    row_end <- row_start + input_nrows[i] - 1L
    nm <- if (nzchar(input_names[i])) input_names[i] else paste0("input", i)
    W_list[[nm]] <- W_full[row_start:row_end, , drop = FALSE]
    row_start <- row_end + 1L
  }

  # --- Package result ---
  layer_result <- list(
    W = W_list,
    d = d_vec,
    H = H_mat,
    iterations = result@misc$iter,
    loss = result@misc$train_loss,
    converged = result@misc$converged
  )

  layer_name <- if (!is.null(layer$name)) layer$name else "L1"

  structure(list(
    layers = setNames(list(layer_result), layer_name),
    config = net$config,
    n_layers = 1L,
    multi_modal = TRUE,
    input_names = names(W_list),
    total_iterations = result@misc$iter,
    total_loss = result@misc$train_loss,
    converged = result@misc$converged
  ), class = "factor_net_result")
}


# ============================================================================
# Deep NMF: R-level outer ALS across layers
# ============================================================================

.fn_fit_deep <- function(net, logger = NULL) {
  layers <- net$layers
  config <- net$config
  n_layers <- length(layers)
  maxit <- config$maxit
  tol <- config$tol
  verbose <- config$verbose

  # --- Detect conditioning nodes between layers ---
  # If layer$input is a "condition" node, extract Z for appending to H
  condition_Z <- vector("list", n_layers)
  for (i in seq_len(n_layers)) {
    if (layers[[i]]$input$type == "condition") {
      condition_Z[[i]] <- layers[[i]]$input$Z
    }
  }

  # Resolve original data from the first layer's input chain
  first_input <- layers[[1]]$input
  if (first_input$type == "shared") {
    stop("Deep NMF with factor_shared() requires single-layer; ",
         "use factor_shared() only on the first layer, or restructure the graph")
  }
  original_data <- .fn_resolve_data(first_input)
  n_samples <- ncol(original_data)

  # --- Helper: orient conditioning Z to p x n ---
  .orient_Z <- function(Z, n) {
    Z <- as.matrix(Z)
    if (ncol(Z) == n) return(Z)       # already p x n
    if (nrow(Z) == n) return(t(Z))    # n x p → p x n
    stop(sprintf("Conditioning matrix Z (%dx%d) incompatible with n=%d samples",
                 nrow(Z), ncol(Z), n))
  }

  # --- Build layer name → index map ---
  layer_name_map <- list()
  for (i in seq_len(n_layers)) {
    nm <- if (!is.null(layers[[i]]$name)) layers[[i]]$name else paste0("L", i)
    layer_name_map[[nm]] <- i
  }

  # --- Helper: find layer index for a node ---
  .find_layer_idx <- function(node) {
    for (j in seq_len(n_layers)) {
      if (identical(node, layers[[j]])) return(j)
    }
    NULL
  }

  # --- Helper: compute effective input for layer i ---
  .effective_input <- function(i, layer_states) {
    layer <- layers[[i]]
    input_node <- layer$input

    # Walk through condition nodes to find the actual source
    cond_node <- NULL
    while (input_node$type == "condition") {
      cond_node <- input_node
      input_node <- input_node$input
    }

    if (input_node$type == "input") {
      # First layer — use original data
      input <- original_data
    } else if (input_node$type == "concat") {
      # Merge: row-bind t(H) from each branch
      branch_inputs <- lapply(input_node$inputs, function(branch) {
        idx <- .find_layer_idx(branch)
        if (is.null(idx)) stop("concat: cannot find branch layer in network")
        t(layer_states[[idx]]$H)  # n x k_branch
      })
      input <- do.call(cbind, branch_inputs)  # n x (k1 + k2 + ...)
    } else if (input_node$type == "add") {
      # Element-wise add: sum t(H) from branches (same k required)
      branch_Hs <- lapply(input_node$inputs, function(branch) {
        idx <- .find_layer_idx(branch)
        if (is.null(idx)) stop("add: cannot find branch layer in network")
        layer_states[[idx]]$H  # k x n
      })
      summed_H <- Reduce(`+`, branch_Hs)
      input <- t(summed_H)  # n x k
    } else {
      # Sequential: previous layer in chain  
      # Find which layer this node is
      idx <- .find_layer_idx(input_node)
      if (is.null(idx)) {
        # Fallback: use previous layer
        if (i <= 1L) stop("Cannot resolve input for first layer")
        idx <- i - 1L
      }
      input <- t(layer_states[[idx]]$H)  # n x k
    }

    # Apply conditioning if present
    if (!is.null(condition_Z[[i]])) {
      Z_oriented <- .orient_Z(condition_Z[[i]], nrow(input))
      input <- cbind(input, t(Z_oriented))
    }

    input
  }

  # --- Initialize all layers ---
  layer_states <- vector("list", n_layers)
  init_maxit <- min(10L, maxit)
  seed_base <- if (!is.null(config$seed)) config$seed else sample.int(.Machine$integer.max, 1)

  for (i in seq_len(n_layers)) {
    layer <- layers[[i]]
    current_input <- .effective_input(i, layer_states)

    init_args <- .fn_build_nmf_args(current_input, layer, config)
    init_args$maxit <- init_maxit
    init_args$seed <- seed_base + i - 1L
    init_args$sort_model <- FALSE

    init_result <- do.call(nmf, init_args)
    layer_states[[i]] <- list(
      W = init_result@w,
      d = init_result@d,
      H = init_result@h
    )
  }

  # --- Outer ALS loop ---
  prev_loss <- Inf
  total_iter <- 0L
  converged <- FALSE

  for (outer_iter in seq_len(maxit)) {
    # Resolve reference guides before each outer iteration
    resolved_guides <- .fn_resolve_reference_guides(layers, layer_states, config, outer_iter)

    for (i in seq_len(n_layers)) {
      layer <- layers[[i]]
      effective_input <- .effective_input(i, layer_states)

      update_args <- .fn_build_nmf_args(effective_input, layer, config)
      update_args$maxit <- 1L
      update_args$sort_model <- FALSE
      update_args$tol <- 0

      # Inject resolved reference guides for this layer
      if (!is.null(resolved_guides[[i]])) {
        existing <- if (!is.null(update_args$guides)) update_args$guides else list()
        update_args$guides <- c(existing, resolved_guides[[i]])
      }

      # Warm-start with current W
      update_args$seed <- layer_states[[i]]$W

      update_result <- do.call(nmf, update_args)
      layer_states[[i]] <- list(
        W = update_result@w,
        d = update_result@d,
        H = update_result@h
      )
    }

    total_iter <- total_iter + 1L

    # Compute per-layer reconstruction loss (works for any topology)
    per_layer_loss <- numeric(n_layers)
    for (i in seq_len(n_layers)) {
      eff <- .effective_input(i, layer_states)
      s <- layer_states[[i]]
      recon <- s$W %*% diag(s$d) %*% s$H
      per_layer_loss[i] <- sum((as.matrix(eff) - recon)^2) / (nrow(eff) * ncol(eff))
    }
    current_loss <- sum(per_layer_loss)

    # Log snapshot
    if (!is.null(logger)) {
      logger <- .log_snapshot(logger, outer_iter, layer_states, layers,
                              per_layer_loss)
    }

    if (verbose) {
      cat(sprintf("  outer iter %d: loss = %g\n", outer_iter, current_loss))
    }

    if (is.finite(prev_loss)) {
      rel_change <- abs(prev_loss - current_loss) / (abs(prev_loss) + 1e-15)
      if (rel_change < tol) {
        converged <- TRUE
        break
      }
    }
    prev_loss <- current_loss
  }

  # --- Package results ---
  layer_results <- list()
  for (i in seq_len(n_layers)) {
    nm <- if (!is.null(layers[[i]]$name)) layers[[i]]$name else paste0("L", i)
    layer_results[[nm]] <- layer_states[[i]]
    layer_results[[nm]]$iterations <- total_iter
    layer_results[[nm]]$loss <- current_loss
    layer_results[[nm]]$converged <- converged
  }

  res <- structure(list(
    layers = layer_results,
    config = net$config,
    n_layers = n_layers,
    multi_modal = FALSE,
    total_iterations = total_iter,
    total_loss = current_loss,
    converged = converged
  ), class = "factor_net_result")
  if (!is.null(logger)) res$logger <- logger
  res
}


# ============================================================================
# Internal: resolve reference guides to current layer matrices
# ============================================================================

.fn_resolve_reference_guides <- function(layers, layer_states, config, outer_iter = 0L) {
  n <- length(layers)
  resolved <- vector("list", n)
  layer_names <- vapply(seq_len(n), function(i)
    if (!is.null(layers[[i]]$name)) layers[[i]]$name else paste0("L", i),
    character(1))

  for (i in seq_len(n)) {
    layer <- layers[[i]]
    ref_guides <- list()

    .check_guides <- function(fc, side) {
      if (is.null(fc) || is.null(fc$guide)) return()
      for (g in fc$guide) {
        if (g$type == "reference") {
          idx <- match(g$layer_name, layer_names)
          if (is.na(idx))
            stop(sprintf("guide_reference: layer '%s' not found in network", g$layer_name))
          target <- if (g$side == "H") layer_states[[idx]]$H else layer_states[[idx]]$W
          ref_guides[[length(ref_guides) + 1L]] <<- guide_external(
            target = target, lambda = g$lambda, side = side)
        }
        if (g$type == "callback") {
          factor <- if (side == "H") layer_states[[i]]$H else layer_states[[i]]$W
          iter <- outer_iter
          target <- g$fn(factor, iter)
          ref_guides[[length(ref_guides) + 1L]] <<- guide_external(
            target = target, lambda = g$lambda, side = side)
        }
      }
    }

    .check_guides(layer$W_config, "W")
    .check_guides(layer$H_config, "H")

    if (length(ref_guides) > 0) resolved[[i]] <- ref_guides
  }
  resolved
}


# ============================================================================
# Internal: reconstruct from deep factorization chain
# ============================================================================

.fn_deep_reconstruct <- function(layer_states) {
  n <- length(layer_states)
  if (n == 1) {
    s <- layer_states[[1]]
    return(s$W %*% diag(s$d) %*% s$H)
  }

  # Each layer i factorizes t(H_{i-1}) ≈ W_i * d_i * H_i
  # So H_{i-1} = t(W_i * d_i * H_i)
  # Walk backward from deepest layer to reconstruct H_1, then X = W_1 * d_1 * H_1
  current_H <- layer_states[[n]]$H
  for (i in rev(seq(2, n))) {
    s <- layer_states[[i]]
    current_H <- t(s$W %*% diag(s$d) %*% current_H)
    # If this layer had conditioning, current_H has extra rows from Z.
    # Strip to match the previous layer's H dimensions.
    target_k <- nrow(layer_states[[i - 1]]$H)
    if (nrow(current_H) > target_k) {
      current_H <- current_H[seq_len(target_k), , drop = FALSE]
    }
  }
  s <- layer_states[[1]]
  s$W %*% diag(s$d) %*% current_H
}


# ============================================================================
# Internal: resolve data from input node chain (extended for deep)
# ============================================================================

.fn_resolve_data <- function(node) {
  if (node$type == "input") return(node$data)
  if (node$type == "condition") {
    # Conditioning is transparent for data resolution — the conditioning
    # matrix Z is applied at the layer boundary (in .fn_fit_deep)
    return(.fn_resolve_data(node$input))
  }
  if (node$type %in% c("nmf_layer", "svd_layer")) {
    return(.fn_resolve_data(node$input))
  }
  if (node$type == "shared") {
    stop("Cannot resolve single data matrix from shared node; ",
         "use .fn_resolve_data on individual inputs")
  }
  stop("Cannot resolve data from node type: ", node$type)
}


# ============================================================================
# Internal: build nmf() argument list from config hierarchy
# ============================================================================

.fn_build_nmf_args <- function(data, layer, config) {
  # Start with global config
  args <- list(
    data = data,
    k = layer$k,
    tol = config$tol,
    maxit = config$maxit,
    verbose = config$verbose,
    threads = config$threads,
    loss = config$loss,
    norm = config$norm,
    resource = config$resource,
    sort_model = TRUE
  )

  if (!is.null(config$seed)) args$seed <- config$seed

  # Solver mapping
  if (config$solver == "cd") args$solver <- "cd"
  else if (config$solver == "cholesky") args$solver <- "cholesky"
  else args$solver <- "auto"

  # Resolve per-factor config: layer defaults < W()/H() overrides
  ld <- layer$layer_defaults
  wc <- layer$W_config
  hc <- layer$H_config

  # L1: c(w, h) — layer default, then factor override
  w_L1 <- if (!is.null(wc) && !is.null(wc$L1)) wc$L1 else ld$L1
  h_L1 <- if (!is.null(hc) && !is.null(hc$L1)) hc$L1 else ld$L1
  args$L1 <- c(w_L1, h_L1)

  # L2
  w_L2 <- if (!is.null(wc) && !is.null(wc$L2)) wc$L2 else ld$L2
  h_L2 <- if (!is.null(hc) && !is.null(hc$L2)) hc$L2 else ld$L2
  args$L2 <- c(w_L2, h_L2)

  # L21
  w_L21 <- if (!is.null(wc) && !is.null(wc$L21)) wc$L21 else ld$L21
  h_L21 <- if (!is.null(hc) && !is.null(hc$L21)) hc$L21 else ld$L21
  args$L21 <- c(w_L21, h_L21)

  # Angular
  w_ang <- if (!is.null(wc) && !is.null(wc$angular)) wc$angular else ld$angular
  h_ang <- if (!is.null(hc) && !is.null(hc$angular)) hc$angular else ld$angular
  args$angular <- c(w_ang, h_ang)

  # Upper bound
  w_ub <- if (!is.null(wc) && !is.null(wc$upper_bound)) wc$upper_bound else ld$upper_bound
  h_ub <- if (!is.null(hc) && !is.null(hc$upper_bound)) hc$upper_bound else ld$upper_bound
  args$upper_bound <- c(w_ub, h_ub)

  # Nonneg: svd_layer defaults to FALSE, nmf_layer defaults to TRUE
  layer_nonneg_default <- (layer$type != "svd_layer")
  w_nn <- if (!is.null(wc) && !is.null(wc$nonneg)) wc$nonneg else layer_nonneg_default
  h_nn <- if (!is.null(hc) && !is.null(hc$nonneg)) hc$nonneg else layer_nonneg_default
  args$nonneg <- c(w_nn, h_nn)

  # Graphs
  if (!is.null(wc) && !is.null(wc$graph)) {
    args$graph_W <- wc$graph
    gl_w <- if (!is.null(wc$graph_lambda)) wc$graph_lambda else 0
  } else {
    gl_w <- 0
  }
  if (!is.null(hc) && !is.null(hc$graph)) {
    args$graph_H <- hc$graph
    gl_h <- if (!is.null(hc$graph_lambda)) hc$graph_lambda else 0
  } else {
    gl_h <- 0
  }
  args$graph_lambda <- c(gl_w, gl_h)

  # Guides: collect from W_config and H_config
  guides <- list()
  if (!is.null(wc) && !is.null(wc$guide)) {
    for (g in wc$guide) {
      g$side <- "W"
      guides <- c(guides, list(g))
    }
  }
  if (!is.null(hc) && !is.null(hc$guide)) {
    for (g in hc$guide) {
      g$side <- "H"
      guides <- c(guides, list(g))
    }
  }
  if (length(guides) > 0) args$guides <- guides

  args
}


# ============================================================================
# Internal: wrap nmf result as factor_net_result
# ============================================================================

.fn_wrap_single_result <- function(nmf_result, layer, net) {
  layer_result <- list(
    W = nmf_result@w,
    d = nmf_result@d,
    H = nmf_result@h,
    iterations = nmf_result@misc$iter,
    loss = nmf_result@misc$train_loss,
    converged = nmf_result@misc$converged
  )

  layer_name <- if (!is.null(layer$name)) layer$name else "L1"

  structure(list(
    layers = setNames(list(layer_result), layer_name),
    config = net$config,
    n_layers = 1L,
    multi_modal = FALSE,
    total_iterations = nmf_result@misc$iter,
    total_loss = nmf_result@misc$train_loss,
    converged = nmf_result@misc$converged
  ), class = "factor_net_result")
}


# ============================================================================
# Result accessors
# ============================================================================

#' Project new data through a trained factor network
#'
#' For single-layer results, projects new samples using the trained W
#' and d. For multi-layer results, chains through each layer.
#'
#' @param object A \code{factor_net_result} from \code{fit()}.
#' @param newdata A matrix (features x samples) to project.
#' @param ... Additional arguments (currently unused).
#' @return For single-layer: a matrix (k x n_new). For multi-layer: a
#'   list of H matrices, one per layer.
#' @seealso \code{\link{fit}}, \code{\link{factor_net}}
#' @method predict factor_net_result
#' @export
predict.factor_net_result <- function(object, newdata, ...) {
  layers <- .subset2(object, "layers")
  layer_list <- as.list(layers)
  n_layers <- length(layer_list)

  if (n_layers == 1L) {
    lr <- layer_list[[1]]
    W <- lr$W
    d <- lr$d
    # For multi-modal, W is a list — use the first modality's dimensions
    if (is.list(W)) {
      stop("predict() for multi-modal results requires matching modality structure; ",
           "concatenate new data in the same order as training inputs")
    }
    # H_new = diag(1/d) * (W^T W)^{-1} W^T X_new
    # Or equivalently, use the predict method from nmf
    model <- new("nmf", w = W, d = d, h = lr$H,
                 misc = list(iter = lr$iterations))
    return(predict(model, newdata)@h)
  }

  # Multi-layer: chain through layers
  current <- newdata
  results <- vector("list", n_layers)
  for (i in seq_len(n_layers)) {
    lr <- layer_list[[i]]
    W <- lr$W; d <- lr$d; H <- lr$H
    model <- new("nmf", w = W, d = d, h = H,
                 misc = list(iter = lr$iterations))
    H_new <- predict(model, current)@h
    results[[i]] <- H_new
    current <- t(H_new)  # n x k for next layer
  }
  names(results) <- names(layers)
  results
}

#' Access layer results by name
#' @param x A \code{factor_net_result} object.
#' @param name Layer name (e.g. \code{"L1"}).
#' @return The layer result list, or the named field if not a layer name.
#' @seealso \code{\link{factor_net}}, \code{\link{fit}}
#' @export
`$.factor_net_result` <- function(x, name) {
  # Allow direct access to layer results by name: result$L1
  layers <- .subset2(x, "layers")
  if (name %in% names(layers)) return(layers[[name]])
  .subset2(x, name)
}

#' Print a factor_net_result
#' @param x A \code{factor_net_result} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{factor_net}}, \code{\link{fit}}, \code{\link{summary.factor_net_result}}
#' @method print factor_net_result
#' @export
print.factor_net_result <- function(x, ...) {
  cat(sprintf("factor_net_result: %d layer(s)", x$n_layers))
  if (x$multi_modal) cat(" [multi-modal]")
  cat("\n")
  for (nm in names(x$layers)) {
    lr <- x$layers[[nm]]
    if (is.list(lr$W)) {
      # Multi-modal: W is a named list
      w_desc <- paste(vapply(names(lr$W), function(n)
        sprintf("%s(%dx%d)", n, nrow(lr$W[[n]]), ncol(lr$W[[n]])),
        character(1)), collapse = " + ")
      cat(sprintf("  %s: W=[%s], d[%d], H(%dx%d)\n",
                  nm, w_desc, length(lr$d),
                  if (is.matrix(lr$H)) nrow(lr$H) else length(lr$d),
                  if (is.matrix(lr$H)) ncol(lr$H) else 0L))
    } else {
      cat(sprintf("  %s: W(%dx%d), d[%d], H(%dx%d)\n",
                  nm,
                  nrow(lr$W), ncol(lr$W),
                  length(lr$d),
                  nrow(lr$H), ncol(lr$H)))
    }
  }
  cat(sprintf("  loss: %g, iter: %d, converged: %s\n",
              if (is.null(x$total_loss)) NA_real_ else x$total_loss,
              if (is.null(x$total_iterations)) 0L else x$total_iterations,
              if (isTRUE(x$converged)) "yes" else "no"))
  invisible(x)
}

#' Summarize a factor_net_result
#' @param object A \code{factor_net_result} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{object}.
#' @seealso \code{\link{factor_net}}, \code{\link{fit}}, \code{\link{print.factor_net_result}}
#' @method summary factor_net_result
#' @export
summary.factor_net_result <- function(object, ...) {
  cat("Factor Network Result Summary\n")
  cat(sprintf("  Layers: %d\n", object$n_layers))
  cat(sprintf("  Multi-modal: %s\n", object$multi_modal))
  cat(sprintf("  Total iterations: %d\n", if (is.null(object$total_iterations)) 0L else object$total_iterations))
  cat(sprintf("  Final loss: %g\n", if (is.null(object$total_loss)) NA_real_ else object$total_loss))
  cat(sprintf("  Converged: %s\n", if (isTRUE(object$converged)) "yes" else "no"))
  for (nm in names(object$layers)) {
    lr <- object$layers[[nm]]
    cat(sprintf("\n  Layer '%s':\n", nm))
    cat(sprintf("    k = %d\n", length(lr$d)))
    cat(sprintf("    d = [%s]\n",
                paste(round(head(lr$d, 5), 2), collapse = ", ")))
  }
  invisible(object)
}


# ============================================================================
# C++ graph dispatcher helpers
# ============================================================================

#' Check if any layer uses R-side callback or reference guides
#' @keywords internal
.fn_has_r_callbacks <- function(layers) {
  for (layer in layers) {
    for (fc in list(layer$W_config, layer$H_config)) {
      if (!is.null(fc) && !is.null(fc$guide)) {
        for (g in fc$guide) {
          if (g$type %in% c("callback", "reference")) return(TRUE)
        }
      }
    }
  }
  FALSE
}

#' Build a flat descriptor for Rcpp_factor_net_fit()
#' @keywords internal
.fn_build_descriptor <- function(net) {
  layers <- net$layers
  config <- net$config
  input_nodes <- net$input_nodes

  # --- Solver mode ---
  # IRLS losses (anything non-MSE) require CD solver; cholesky is only valid for MSE.
  irls_losses <- c("mae", "huber", "kl", "gp", "nb", "gamma", "inverse_gaussian", "tweedie")
  solver_mode <- if (config$solver == "cd") 0L
                 else if (config$solver == "cholesky") 1L
                 else if (config$loss %in% irls_losses) 0L
                 else 1L  # auto defaults to cholesky for MSE

  # --- Build config ---
  cfg_list <- list(
    maxit = config$maxit,
    tol = config$tol,
    verbose = config$verbose,
    threads = config$threads,
    seed = if (!is.null(config$seed)) as.integer(config$seed) else 0L,
    solver_mode = solver_mode,
    resource = config$resource,
    norm = config$norm,
    loss = config$loss,
    huber_delta = 1.0,
    holdout_fraction = if (!is.null(config$test_fraction)) config$test_fraction
                       else if (!is.null(config$holdout_fraction)) config$holdout_fraction else 0,
    cv_seed = if (!is.null(config$cv_seed)) as.integer(config$cv_seed) else 0L,
    mask_zeros = if (!is.null(config$mask_zeros)) config$mask_zeros else FALSE,
    cv_patience = if (!is.null(config$patience)) as.integer(config$patience)
                  else if (!is.null(config$cv_patience)) as.integer(config$cv_patience) else 5L
  )

  # --- Build input descriptors ---
  inputs <- lapply(input_nodes, function(inp) {
    res <- list(name = inp$name)
    if (is.character(inp$data) && length(inp$data) == 1 &&
        grepl("\\.spz$", inp$data, ignore.case = TRUE)) {
      res$spz_path <- inp$data
    } else {
      res$data <- inp$data
    }
    res
  })

  # --- Build layer name map ---
  layer_names <- vapply(seq_along(layers), function(i) {
    if (!is.null(layers[[i]]$name)) layers[[i]]$name else paste0("L", i)
  }, character(1))

  # --- Build condition and layer descriptors ---
  conditions <- list()
  layer_descs <- list()

  for (i in seq_along(layers)) {
    layer <- layers[[i]]
    lname <- layer_names[i]
    ltype <- sub("_layer$", "", layer$type)

    # Resolve input reference
    input_ref <- .fn_resolve_input_ref(layer$input, layer_names, layers,
                                        input_nodes, conditions)

    # If the resolution created conditions, they're added to the list
    if (!is.null(attr(input_ref, "conditions"))) {
      conditions <- c(conditions, attr(input_ref, "conditions"))
    }

    # Build factor configs
    ld <- layer$layer_defaults
    wc <- layer$W_config
    hc <- layer$H_config
    default_nn <- (ltype == "nmf")

    w_cfg <- list(
      L1 = if (!is.null(wc) && !is.null(wc$L1)) wc$L1 else ld$L1,
      L2 = if (!is.null(wc) && !is.null(wc$L2)) wc$L2 else ld$L2,
      L21 = if (!is.null(wc) && !is.null(wc$L21)) wc$L21 else ld$L21,
      angular = if (!is.null(wc) && !is.null(wc$angular)) wc$angular else ld$angular,
      upper_bound = if (!is.null(wc) && !is.null(wc$upper_bound)) wc$upper_bound else ld$upper_bound,
      nonneg = if (!is.null(wc) && !is.null(wc$nonneg)) wc$nonneg else default_nn,
      graph_lambda = if (!is.null(wc) && !is.null(wc$graph_lambda)) wc$graph_lambda else 0
    )
    if (!is.null(wc) && !is.null(wc$graph)) w_cfg$graph <- wc$graph

    h_cfg <- list(
      L1 = if (!is.null(hc) && !is.null(hc$L1)) hc$L1 else ld$L1,
      L2 = if (!is.null(hc) && !is.null(hc$L2)) hc$L2 else ld$L2,
      L21 = if (!is.null(hc) && !is.null(hc$L21)) hc$L21 else ld$L21,
      angular = if (!is.null(hc) && !is.null(hc$angular)) hc$angular else ld$angular,
      upper_bound = if (!is.null(hc) && !is.null(hc$upper_bound)) hc$upper_bound else ld$upper_bound,
      nonneg = if (!is.null(hc) && !is.null(hc$nonneg)) hc$nonneg else default_nn,
      graph_lambda = if (!is.null(hc) && !is.null(hc$graph_lambda)) hc$graph_lambda else 0
    )
    if (!is.null(hc) && !is.null(hc$graph)) h_cfg$graph <- hc$graph

    layer_descs[[i]] <- list(
      name = lname,
      type = ltype,
      k = layer$k,
      input_ref = input_ref,
      W_config = w_cfg,
      H_config = h_cfg
    )
  }

  # --- Build guides list (classifier + external only) ---
  guides <- list()
  for (i in seq_along(layers)) {
    layer <- layers[[i]]
    for (side_info in list(
      list(fc = layer$W_config, side = "W"),
      list(fc = layer$H_config, side = "H"))) {
      fc <- side_info$fc
      if (!is.null(fc) && !is.null(fc$guide)) {
        for (g in fc$guide) {
          if (g$type == "classifier") {
            guides[[length(guides) + 1L]] <- list(
              type = "classifier", lambda = g$lambda,
              side = side_info$side, layer_idx = i - 1L,
              labels = g$labels)
          } else if (g$type == "external") {
            guides[[length(guides) + 1L]] <- list(
              type = "external", lambda = g$lambda,
              side = side_info$side, layer_idx = i - 1L,
              target = g$target)
          }
        }
      }
    }
  }

  desc <- list(
    inputs = inputs,
    layers = layer_descs,
    config = cfg_list
  )
  if (length(conditions) > 0) desc$conditions <- conditions
  if (length(guides) > 0) desc$guides <- guides

  # --- Generate W_init for the first layer (match nmf() R-RNG behavior) ---
  # nmf() uses set.seed(seed); runif(m*k) to initialize W, then passes
  # it to C++. We must do the same here so the graph path matches.
  if (length(layer_descs) >= 1 && !is.null(config$seed)) {
    # Determine m from the first input (after multi-modal concat if needed)
    first_ref <- layer_descs[[1]]$input_ref
    if (grepl("^input:", first_ref)) {
      inp_name <- sub("^input:", "", first_ref)
      inp_node <- NULL
      for (nd in input_nodes) {
        if (nd$name == inp_name) { inp_node <- nd; break }
      }
      if (!is.null(inp_node) && !is.character(inp_node$data)) {
        m <- nrow(inp_node$data)
      } else {
        m <- NULL
      }
    } else if (grepl("^shared:", first_ref)) {
      # Multi-modal: m = sum of all input row counts
      inp_names <- strsplit(sub("^shared:", "", first_ref), ",")[[1]]
      m <- 0L
      for (nm in inp_names) {
        for (nd in input_nodes) {
          if (nd$name == nm) { m <- m + nrow(nd$data); break }
        }
      }
    } else {
      m <- NULL
    }

    if (!is.null(m)) {
      k <- layer_descs[[1]]$k
      set.seed(as.integer(config$seed))
      desc$w_init <- matrix(runif(m * k), m, k)
    }
  }

  desc
}

#' Resolve a node to an input_ref string for the C++ descriptor
#' @keywords internal
.fn_resolve_input_ref <- function(node, layer_names, layers, input_nodes,
                                   conditions) {
  new_conditions <- NULL

  if (node$type == "input") {
    return(paste0("input:", node$name))
  }

  if (node$type == "condition") {
    # Create a condition descriptor and return the reference
    cond_name <- paste0("cond_", length(conditions) + 1L)
    upstream_ref <- .fn_resolve_input_ref(node$input, layer_names, layers,
                                           input_nodes, conditions)
    cond <- list(
      name = cond_name,
      input_ref = upstream_ref,
      Z = node$Z
    )
    new_conditions <- list(cond)
    ref <- paste0("cond:", cond_name)
    attr(ref, "conditions") <- new_conditions
    return(ref)
  }

  if (node$type == "shared") {
    names_vec <- vapply(node$inputs, function(inp) inp$name, character(1))
    return(paste0("shared:", paste(names_vec, collapse = ",")))
  }

  if (node$type == "concat") {
    branch_names <- vapply(node$inputs, function(br) {
      idx <- .fn_find_layer_idx(br, layers)
      if (is.null(idx)) stop("concat: branch not found in layers")
      layer_names[idx]
    }, character(1))
    return(paste0("concat:", paste(branch_names, collapse = ",")))
  }

  if (node$type == "add") {
    branch_names <- vapply(node$inputs, function(br) {
      idx <- .fn_find_layer_idx(br, layers)
      if (is.null(idx)) stop("add: branch not found in layers")
      layer_names[idx]
    }, character(1))
    return(paste0("add:", paste(branch_names, collapse = ",")))
  }

  if (node$type %in% c("nmf_layer", "svd_layer")) {
    idx <- .fn_find_layer_idx(node, layers)
    if (!is.null(idx)) return(paste0("layer:", layer_names[idx]))
  }

  stop("Cannot resolve input_ref for node type: ", node$type)
}

#' Find the index of a node in the layers list
#' @keywords internal
.fn_find_layer_idx <- function(node, layers) {
  for (j in seq_along(layers)) {
    if (identical(node, layers[[j]])) return(j)
  }
  NULL
}

#' Wrap C++ graph result into factor_net_result
#' @keywords internal
.fn_wrap_graph_result <- function(raw, net) {
  layers <- raw$layers
  layer_names <- names(layers)

  # Wrap each layer
  layer_results <- list()
  for (nm in layer_names) {
    lr <- layers[[nm]]
    layer_result <- list(
      W = lr$W,
      d = as.numeric(lr$d),
      H = lr$H,
      iterations = if (length(lr$iterations)) lr$iterations else 0L,
      loss = if (length(lr$loss)) lr$loss else NA_real_,
      test_loss = if (length(lr$test_loss)) lr$test_loss else NA_real_,
      best_test_loss = if (length(lr$best_test_loss)) lr$best_test_loss else NA_real_,
      converged = if (length(lr$converged)) lr$converged else FALSE
    )
    # Handle multi-modal W splits
    if (!is.null(lr$W_splits)) {
      layer_result$W <- lr$W_splits  # W is a named list
    }
    layer_results[[nm]] <- layer_result
  }

  is_mm <- isTRUE(raw$multi_modal)

  res <- structure(list(
    layers = layer_results,
    config = net$config,
    n_layers = length(layer_results),
    multi_modal = is_mm,
    total_iterations = if (length(raw$total_iterations)) raw$total_iterations else 0L,
    total_loss = if (length(raw$total_loss)) raw$total_loss else NA_real_,
    converged = if (length(raw$converged)) raw$converged else FALSE
  ), class = "factor_net_result")

  if (is_mm) {
    # Collect input names from the multi-modal result
    first_lr <- layer_results[[1]]
    if (is.list(first_lr$W)) {
      res$input_names <- names(first_lr$W)
    }
  }

  res
}
