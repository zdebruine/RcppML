# factor_net.R
#
# Composable factorization network API. Supports arbitrary DAGs of
# factorization layers (NMF, SVD) with per-factor configuration,
# multi-modal shared inputs, and conditioning.
#
# Key concepts:
#   fn_node     — graph node (input, layer, shared, condition)
#   fn_config   — per-factor config (regularization, guides)
#   factor_net  — compiled graph ready to fit
#
# Execution delegates to the existing nmf()/svd() solvers.

# ============================================================================
# Per-factor config constructors: W() and H()
# ============================================================================

#' Per-factor configuration for factorization layers
#'
#' Use \code{W()} and \code{H()} inside \code{nmf_layer()} or
#' \code{svd_layer()} to set factor-specific regularization, constraints,
#' and guides. Parameters not set inherit from the layer defaults.
#'
#' @param L1 L1 (lasso) penalty. Default 0.
#' @param L2 L2 (ridge) penalty. Default 0.
#' @param L21 Group sparsity penalty. Default 0.
#' @param angular Orthogonality penalty. Default 0.
#' @param upper_bound Box constraint upper bound (0 = none). Default 0.
#' @param nonneg Non-negativity constraint. Default TRUE for NMF, FALSE for SVD.
#' @param guide An \code{nmf_guide} or list of guides for this factor.
#' @param graph Sparse graph Laplacian matrix (or NULL).
#' @param graph_lambda Graph regularization strength. Default 0.
#' @return An \code{fn_factor_config} object.
#' @seealso \code{\link{factor_config}}, \code{\link{nmf_layer}}, \code{\link{factor_net}}
#' @examples
#' # Per-factor config for W with L1 sparsity
#' w_cfg <- W(L1 = 0.1, nonneg = TRUE)
#' h_cfg <- H(L2 = 0.01)
#' @export
W <- function(L1 = NULL, L2 = NULL, L21 = NULL, angular = NULL,
              upper_bound = NULL, nonneg = NULL,
              guide = NULL, graph = NULL, graph_lambda = NULL) {
  .fn_factor_config("W", L1, L2, L21, angular, upper_bound, nonneg,
                    guide, graph, graph_lambda)
}

#' @rdname W
#' @export
H <- function(L1 = NULL, L2 = NULL, L21 = NULL, angular = NULL,
              upper_bound = NULL, nonneg = NULL,
              guide = NULL, graph = NULL, graph_lambda = NULL) {
  .fn_factor_config("H", L1, L2, L21, angular, upper_bound, nonneg,
                    guide, graph, graph_lambda)
}

.fn_factor_config <- function(side, L1, L2, L21, angular, upper_bound,
                              nonneg, guide, graph, graph_lambda) {
  if (!is.null(guide)) {
    if (inherits(guide, "nmf_guide")) guide <- list(guide)
    if (!is.list(guide) || !all(vapply(guide, inherits, logical(1), "nmf_guide")))
      stop("'guide' must be an nmf_guide object or a list of nmf_guide objects")
  }
  structure(list(
    side = side,
    L1 = L1, L2 = L2, L21 = L21, angular = angular,
    upper_bound = upper_bound, nonneg = nonneg,
    guide = guide, graph = graph, graph_lambda = graph_lambda
  ), class = "fn_factor_config")
}


# ============================================================================
# Global network config
# ============================================================================

#' Global configuration for a factorization network
#'
#' Sets network-wide defaults. Layer-level and factor-level settings
#' override these where specified.
#'
#' @param maxit Maximum ALS iterations per layer. Default 100.
#' @param tol Convergence tolerance. Default 1e-4.
#' @param loss Loss function: "mse", "mae", "huber", "kl", "gp",
#'   "nb", "gamma", "inverse_gaussian", "tweedie". Default "mse".
#' @param verbose Print per-iteration diagnostics. Default FALSE.
#' @param seed Random seed. Default NULL (auto).
#' @param threads Number of CPU threads (0 = all). Default 0.
#' @param norm Normalization type: "L1", "L2", "none". Default "L1".
#' @param solver NNLS solver: "auto", "cholesky", "cd". Default "auto".
#' @param resource Resource override: "auto", "cpu", "gpu". Default "auto".
#' @param test_fraction Fraction of entries held out for cross-validation
#'   test set. 0 = no CV (standard fit). Default 0.
#' @param cv_seed Separate seed for CV mask (0 = derive from \code{seed}).
#' @param mask_zeros If TRUE, only non-zero entries are in the test set
#'   (suitable for recommendation systems). If FALSE, all entries including
#'   zeros may appear in the test set. Default FALSE.
#' @param patience Number of iterations without test-loss improvement
#'   before early stopping during CV. Default 5.
#' @param holdout_fraction Deprecated. Use \code{test_fraction} instead.
#' @param cv_patience Deprecated. Use \code{patience} instead.
#' @details
#' \strong{Unsupported combinations:}
#' \itemize{
#'   \item Cholesky solver with non-MSE losses (GP, NB, Gamma, etc.) is not
#'     supported. Use \code{solver="cd"} or \code{solver="auto"}.
#'   \item Zero-inflation is only available with GP and NB losses at the layer
#'     level; it cannot be set globally in \code{factor_config()}.
#' }
#' @return An \code{fn_global_config} object.
#' @examples
#' # Default config
#' cfg <- factor_config()
#'
#' # Poisson NMF with cross-validation
#' cfg <- factor_config(loss = "gp", test_fraction = 0.1, maxit = 50)
#' @seealso \code{\link{factor_net}}, \code{\link{nmf_layer}}, \code{\link{fit}}
#' @export
factor_config <- function(maxit = 100, tol = 1e-4,
                          loss = c("mse", "mae", "huber", "kl", "gp",
                                   "nb", "gamma", "inverse_gaussian", "tweedie"),
                          verbose = FALSE, seed = NULL, threads = 0,
                          norm = c("L1", "L2", "none"),
                          solver = c("auto", "cholesky", "cd"),
                          resource = "auto",
                          test_fraction = 0,
                          cv_seed = 0L,
                          mask_zeros = FALSE,
                          patience = 5L,
                          holdout_fraction, cv_patience) {
  # Handle deprecated aliases
  if (!missing(holdout_fraction)) {
    .Deprecated("test_fraction", old = "holdout_fraction")
    if (missing(test_fraction) || test_fraction == 0) test_fraction <- holdout_fraction
  }
  if (!missing(cv_patience)) {
    .Deprecated("patience", old = "cv_patience")
    if (missing(patience) || patience == 5L) patience <- cv_patience
  }
  loss <- match.arg(loss)
  norm <- match.arg(norm)
  solver <- match.arg(solver)
  stopifnot(test_fraction >= 0 && test_fraction < 1)
  structure(list(
    maxit = as.integer(maxit),
    tol = as.double(tol),
    loss = loss,
    verbose = verbose,
    seed = seed,
    threads = as.integer(threads),
    norm = norm,
    solver = solver,
    resource = resource,
    test_fraction = as.double(test_fraction),
    cv_seed = as.integer(cv_seed),
    mask_zeros = as.logical(mask_zeros),
    patience = as.integer(patience)
  ), class = "fn_global_config")
}


# ============================================================================
# Graph node constructors
# ============================================================================

#' Create an input node for a factorization network
#'
#' Wraps a data matrix as a graph input node. The matrix can be dense
#' or sparse (dgCMatrix), or a file path to a .spz file for streaming.
#'
#' @param data A numeric matrix, sparse matrix (dgCMatrix), or character
#'   string path to a .spz file for out-of-core streaming NMF.
#' @param name Optional name for the input (used in multi-modal results).
#' @return An \code{fn_node} object of type "input".
#' @seealso \code{\link{nmf_layer}}, \code{\link{svd_layer}}, \code{\link{factor_net}}
#' @examples
#' data(aml)
#' inp <- factor_input(aml, name = "aml")
#' inp
#' @export
factor_input <- function(data, name = NULL) {
  is_path <- is.character(data) && length(data) == 1 && file.exists(data)
  if (!is_path && !is.matrix(data) && !inherits(data, "dgCMatrix"))
    stop("'data' must be a dense matrix, dgCMatrix, or path to .spz file")
  if (!is_path && (nrow(data) == 0 || ncol(data) == 0))
    stop("Input matrix cannot be empty (0 rows or 0 columns)")
  if (is.null(name)) {
    if (is_path) name <- tools::file_path_sans_ext(basename(data))
    else name <- deparse(substitute(data))
  }
  structure(list(
    type = "input",
    data = data,
    name = name
  ), class = "fn_node")
}

#' Create an NMF factorization layer
#'
#' Creates a non-negative matrix factorization layer that decomposes
#' its input into W * diag(d) * H with non-negativity constraints.
#'
#' When used with \code{|>}, the input node is the first argument:
#' \code{x |> nmf_layer(k = 64)}.
#'
#' Layer-level regularization parameters (L1, L2, etc.) apply to both
#' W and H unless overridden by \code{W()} or \code{H()}.
#'
#' @param input An \code{fn_node} (input, shared, condition, or another layer).
#' @param k Factorization rank.
#' @param L1 L1 penalty (shared by W and H unless overridden). Default 0.
#' @param L2 L2 penalty. Default 0.
#' @param L21 Group sparsity penalty. Default 0.
#' @param angular Orthogonality penalty. Default 0.
#' @param upper_bound Box constraint. Default 0.
#' @param W Optional \code{W()} config to override W-specific settings.
#' @param H Optional \code{H()} config to override H-specific settings.
#' @param name Optional layer name (for results access).
#' @return An \code{fn_node} of type "nmf_layer".
#' @examples
#' data(aml)
#' inp <- factor_input(aml)
#' layer <- nmf_layer(inp, k = 5)
#' layer
#' @seealso \code{\link{svd_layer}}, \code{\link{factor_net}}, \code{\link{factor_input}}
#' @export
nmf_layer <- function(input, k, L1 = 0, L2 = 0, L21 = 0, angular = 0,
                      upper_bound = 0, W = NULL, H = NULL, name = NULL) {
  if (!inherits(input, "fn_node"))
    stop("'input' must be an fn_node (from factor_input, nmf_layer, etc.)")
  if (!is.null(W) && !inherits(W, "fn_factor_config"))
    stop("'W' must be created by W()")
  if (!is.null(H) && !inherits(H, "fn_factor_config"))
    stop("'H' must be created by H()")
  structure(list(
    type = "nmf_layer",
    input = input,
    k = as.integer(k),
    layer_defaults = list(L1 = L1, L2 = L2, L21 = L21, angular = angular,
                          upper_bound = upper_bound),
    W_config = W,
    H_config = H,
    name = name
  ), class = "fn_node")
}

#' Create an SVD/PCA factorization layer
#'
#' Creates an unconstrained (SVD/PCA) factorization layer. No
#' non-negativity constraint; factors are signed.
#'
#' @inheritParams nmf_layer
#' @return An \code{fn_node} of type "svd_layer".
#' @seealso \code{\link{nmf_layer}}, \code{\link{factor_input}}, \code{\link{factor_net}}
#' @examples
#' data(aml)
#' inp <- factor_input(aml)
#' layer <- svd_layer(inp, k = 5)
#' @export
svd_layer <- function(input, k, L1 = 0, L2 = 0, L21 = 0, angular = 0,
                      upper_bound = 0, W = NULL, H = NULL, name = NULL) {
  if (!inherits(input, "fn_node"))
    stop("'input' must be an fn_node")
  structure(list(
    type = "svd_layer",
    input = input,
    k = as.integer(k),
    layer_defaults = list(L1 = L1, L2 = L2, L21 = L21, angular = angular,
                          upper_bound = upper_bound),
    W_config = W,
    H_config = H,
    name = name
  ), class = "fn_node")
}


# ============================================================================
# Merge operations
# ============================================================================

#' Shared factorization across multiple inputs (multi-modal)
#'
#' Creates a node representing shared-H factorization of multiple
#' input matrices. The resulting H is shared across all inputs;
#' each input gets its own W.
#'
#' Execution concatenates inputs row-wise and runs a single NMF:
#' \code{rbind(X1, X2, ...) = rbind(W1, W2, ...) * diag(d) * H}.
#'
#' @param ... Two or more \code{fn_node} objects (inputs or layers).
#' @return An \code{fn_node} of type "shared".
#' @seealso \code{\link{factor_concat}}, \code{\link{factor_add}}, \code{\link{factor_net}}
#' @examples
#' data(aml)
#' # Split into two views
#' inp1 <- factor_input(aml[1:400, ], name = "view1")
#' inp2 <- factor_input(aml[401:824, ], name = "view2")
#' shared <- factor_shared(inp1, inp2)
#' @export
factor_shared <- function(...) {
  inputs <- list(...)
  if (length(inputs) < 2)
    stop("factor_shared() requires at least 2 inputs")
  if (!all(vapply(inputs, inherits, logical(1), "fn_node")))
    stop("All arguments to factor_shared() must be fn_node objects")
  # Extract names from inputs
  input_names <- vapply(inputs, function(x) {
    if (!is.null(x$name)) x$name else ""
  }, character(1))
  structure(list(
    type = "shared",
    inputs = inputs,
    input_names = input_names
  ), class = "fn_node")
}

#' Concatenate H factors from branches (row-bind)
#'
#' Creates a node that row-binds the H factors from multiple branches.
#' Allows combining branches with different ranks into a single
#' higher-dimensional representation: k_out = k_1 + k_2 + ...
#'
#' @param ... Two or more \code{fn_node} objects (layer outputs).
#' @return An \code{fn_node} of type "concat".
#' @seealso \code{\link{factor_shared}}, \code{\link{factor_add}}, \code{\link{factor_net}}
#' @examples
#' data(aml)
#' inp <- factor_input(aml)
#' branch1 <- nmf_layer(inp, k = 3)
#' branch2 <- nmf_layer(inp, k = 5)
#' combined <- factor_concat(branch1, branch2)
#' @export
factor_concat <- function(...) {
  inputs <- list(...)
  if (length(inputs) < 2)
    stop("factor_concat() requires at least 2 inputs")
  if (!all(vapply(inputs, inherits, logical(1), "fn_node")))
    stop("All arguments to factor_concat() must be fn_node objects")
  structure(list(
    type = "concat",
    inputs = inputs
  ), class = "fn_node")
}

#' Element-wise H addition (skip/residual connection)
#'
#' Creates a node that adds H factors element-wise from multiple branches.
#' All branches must have the same rank k.
#'
#' @param ... Two or more \code{fn_node} objects with matching ranks.
#' @return An \code{fn_node} of type "add".
#' @seealso \code{\link{factor_concat}}, \code{\link{factor_shared}}, \code{\link{factor_net}}
#' @examples
#' data(aml)
#' inp <- factor_input(aml)
#' branch1 <- nmf_layer(inp, k = 5)
#' branch2 <- nmf_layer(inp, k = 5)
#' added <- factor_add(branch1, branch2)
#' @export
factor_add <- function(...) {
  inputs <- list(...)
  if (length(inputs) < 2)
    stop("factor_add() requires at least 2 inputs")
  if (!all(vapply(inputs, inherits, logical(1), "fn_node")))
    stop("All arguments to factor_add() must be fn_node objects")
  # Validate that all layer inputs have the same rank k
  ranks <- vapply(inputs, function(x) {
    if (!is.null(x$k)) x$k else NA_integer_
  }, integer(1))
  ranks <- ranks[!is.na(ranks)]
  if (length(ranks) >= 2 && length(unique(ranks)) > 1)
    stop("factor_add() requires all branches to have the same rank k, got: ",
         paste(ranks, collapse = ", "))
  structure(list(
    type = "add",
    inputs = inputs
  ), class = "fn_node")
}

#' Concatenate conditioning metadata to a layer's H
#'
#' Appends columns from a metadata matrix Z to the H factor before
#' passing downstream. The next layer's W learns to factor out the
#' conditioning variables.
#'
#' @param input An \code{fn_node} (typically an nmf_layer output).
#' @param Z Conditioning matrix (n x p) or (p x n). Will be oriented
#'   to match H dimensions.
#' @return An \code{fn_node} of type "condition".
#' @seealso \code{\link{nmf_layer}}, \code{\link{factor_net}}
#' @examples
#' data(aml)
#' inp <- factor_input(aml)
#' layer1 <- nmf_layer(inp, k = 5)
#' # Condition on batch metadata (2 batches)
#' Z <- matrix(rep(c(1, 0, 0, 1), c(60, 75, 60, 75)), nrow = 135, ncol = 2)
#' conditioned <- factor_condition(layer1, Z)
#' @export
factor_condition <- function(input, Z) {
  if (!inherits(input, "fn_node"))
    stop("'input' must be an fn_node")
  if (!is.matrix(Z) && !is.numeric(Z))
    stop("'Z' must be a numeric matrix")
  Z <- as.matrix(Z)
  structure(list(
    type = "condition",
    input = input,
    Z = Z
  ), class = "fn_node")
}


# ============================================================================
# Network compilation
# ============================================================================

#' Compile a factorization network
#'
#' Validates the graph topology, resolves config inheritance, and
#' returns a compiled network ready for \code{fit()}.
#'
#' @param inputs A single \code{fn_node} (input) or a list of input nodes.
#' @param output The output \code{fn_node} (typically a layer).
#' @param config A \code{fn_global_config} from \code{factor_config()}.
#'   Default uses \code{factor_config()} defaults.
#' @return A \code{factor_net} object.
#' @examples
#' data(aml)
#' inp <- factor_input(aml, "aml")
#' out <- nmf_layer(inp, k = 5)
#' net <- factor_net(inp, out, config = factor_config(maxit = 10))
#' net
#' @seealso \code{\link{fit}}, \code{\link{factor_config}}, \code{\link{nmf_layer}},
#'   \code{\link{svd_layer}}, \code{\link{cross_validate_graph}}
#' @export
factor_net <- function(inputs, output, config = factor_config()) {
  # Normalize inputs to a list

  if (inherits(inputs, "fn_node")) inputs <- list(inputs)
  if (!all(vapply(inputs, inherits, logical(1), "fn_node")))
    stop("'inputs' must be fn_node objects")
  if (!inherits(output, "fn_node"))
    stop("'output' must be an fn_node")
  if (!inherits(config, "fn_global_config"))
    stop("'config' must be from factor_config()")

  # Walk the graph to collect layers and validate topology
  layers <- .fn_collect_layers(output)
  input_nodes <- .fn_collect_inputs(output)

  # Validate all declared inputs appear in the graph
  declared_names <- vapply(inputs, function(x) x$name, character(1))
  graph_names <- vapply(input_nodes, function(x) x$name, character(1))
  missing <- setdiff(declared_names, graph_names)
  if (length(missing) > 0)
    stop("Declared inputs not found in graph: ", paste(missing, collapse = ", "))

  # Assign default names to layers
  for (i in seq_along(layers)) {
    if (is.null(layers[[i]]$name))
      layers[[i]]$name <- paste0("L", i)
  }

  structure(list(
    inputs = inputs,
    output = output,
    config = config,
    layers = layers,
    input_nodes = input_nodes,
    n_layers = length(layers)
  ), class = "factor_net")
}


# ============================================================================
# Internal graph traversal helpers
# ============================================================================

.fn_collect_layers <- function(node) {
  if (node$type %in% c("nmf_layer", "svd_layer")) {
    upstream <- .fn_collect_layers(node$input)
    return(c(upstream, list(node)))
  }
  if (node$type %in% c("shared", "concat", "add")) {
    all_up <- do.call(c, lapply(node$inputs, .fn_collect_layers))
    return(all_up)
  }
  if (node$type == "condition") {
    return(.fn_collect_layers(node$input))
  }
  # input node — no layers
  list()
}

.fn_collect_inputs <- function(node) {
  if (node$type == "input") return(list(node))
  if (node$type %in% c("shared", "concat", "add")) {
    return(do.call(c, lapply(node$inputs, .fn_collect_inputs)))
  }
  if (node$type %in% c("nmf_layer", "svd_layer", "condition")) {
    return(.fn_collect_inputs(node$input))
  }
  list()
}


# ============================================================================
# Print methods
# ============================================================================

#' Print an fn_node
#' @param x An \code{fn_node} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{factor_input}}, \code{\link{nmf_layer}}, \code{\link{factor_net}}
#' @method print fn_node
#' @export
print.fn_node <- function(x, ...) {
  cat(sprintf("fn_node: %s", x$type))
  if (!is.null(x$name)) cat(sprintf(" (%s)", x$name))
  if (!is.null(x$k)) cat(sprintf(", k=%d", x$k))
  cat("\n")
  invisible(x)
}

#' Print an fn_factor_config
#' @param x An \code{fn_factor_config} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{W}}, \code{\link{H}}, \code{\link{factor_config}}
#' @method print fn_factor_config
#' @export
print.fn_factor_config <- function(x, ...) {
  cat(sprintf("fn_factor_config: %s-side\n", x$side))
  fields <- c("L1", "L2", "L21", "angular", "upper_bound", "nonneg",
              "graph_lambda")
  for (f in fields) {
    if (!is.null(x[[f]])) cat(sprintf("  %s = %s\n", f, x[[f]]))
  }
  if (!is.null(x$guide)) cat(sprintf("  guides: %d\n", length(x$guide)))
  invisible(x)
}

#' Print an fn_global_config
#' @param x An \code{fn_global_config} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{factor_config}}, \code{\link{factor_net}}
#' @method print fn_global_config
#' @export
print.fn_global_config <- function(x, ...) {
  cat(sprintf("factor_config: maxit=%d, tol=%g, loss=%s, solver=%s\n",
              x$maxit, x$tol, x$loss, x$solver))
  invisible(x)
}

#' Print a factor_net
#' @param x A \code{factor_net} object.
#' @param ... Additional arguments (unused).
#' @return Invisibly returns \code{x}.
#' @seealso \code{\link{factor_net}}, \code{\link{fit}}
#' @method print factor_net
#' @export
print.factor_net <- function(x, ...) {
  cat(sprintf("factor_net: %d layer(s), %d input(s)\n",
              x$n_layers, length(x$input_nodes)))
  for (layer in x$layers) {
    ltype <- sub("_layer$", "", layer$type)
    cat(sprintf("  %s: %s(k=%d)\n",
                if (!is.null(layer$name)) layer$name else "?",
                toupper(ltype), layer$k))
  }
  cat(sprintf("  config: maxit=%d, tol=%g, loss=%s\n",
              x$config$maxit, x$config$tol, x$config$loss))
  invisible(x)
}
