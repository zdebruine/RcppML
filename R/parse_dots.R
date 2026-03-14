# Parse and validate advanced parameters from ... for nmf(), svd(), and nnls()
# These helpers extract known parameters from dots, apply defaults,
# and raise errors for unknown parameters.

#' Parse advanced NMF parameters from dots
#' @param dots list from list(...)
#' @return Named list with all advanced parameters filled to defaults
#' @keywords internal
.parse_nmf_dots <- function(dots) {
  defaults <- list(
    # Regularization
    L21 = c(0, 0),
    angular = c(0, 0),
    upper_bound = c(0, 0),
    graph_W = NULL,
    graph_H = NULL,
    graph_lambda = c(0, 0),
    # Target regularization
    target_H = NULL,
    target_lambda = c(0, 0),
    # Distribution tuning
    dispersion = "per_row",
    theta_init = 0.1,
    theta_max = 5.0,
    theta_min = 0.0,
    nb_size_init = 10.0,
    nb_size_max = 1e6,
    nb_size_min = 0.01,
    gamma_phi_init = 1.0,
    gamma_phi_max = 1e4,
    gamma_phi_min = 1e-6,
    tweedie_power = 1.5,
    irls_max_iter = 5L,
    irls_tol = 1e-4,
    # Huber loss tuning
    huber_delta = 1.0,
    # Zero-inflation
    zi_em_iters = 1L,
    # Solver
    solver = "auto",
    cd_tol = 1e-8,
    cd_maxit = 100L,
    h_init = NULL,
    w_init = NULL,
    # Cross-validation
    cv_seed = NULL,
    patience = 5,
    cv_k_range = c(2, 50),
    track_train_loss = TRUE,
    # Resources & output
    threads = 0,
    resource = "auto",
    norm = "L1",
    sort_model = TRUE,
    # Streaming
    streaming = "auto",
    panel_cols = 0L,
    dispatch = NULL,
    # Callbacks
    on_iteration = NULL,
    profile = FALSE,
    # Convergence mode
    convergence = "loss",
    # Sparse mode (treat zeros as missing)
    sparse = FALSE
  )

  # Published API params that were passed via ... in the old API
  # (sort_model, upper_bound, link_matrix_h, link_h)
  # Keep backward compat for these

  known <- names(defaults)
  unknown <- setdiff(names(dots), known)
  if (length(unknown) > 0) {
    stop("Unknown parameter(s) passed to nmf(): ",
         paste0("'", unknown, "'", collapse = ", "),
         ". See ?nmf for valid parameters.",
         call. = FALSE)
  }

  # Merge user values over defaults
  result <- defaults
  for (nm in intersect(names(dots), known)) {
    result[[nm]] <- dots[[nm]]
  }

  # Expand length-1 to length-2 for paired params
  for (nm in c("L21", "angular", "upper_bound", "graph_lambda")) {
    if (length(result[[nm]]) == 1) result[[nm]] <- rep(result[[nm]], 2)
  }

  # Match string args
  result$dispersion <- match.arg(result$dispersion,
                                  c("per_row", "per_col", "global", "none"))
  result$solver <- match.arg(result$solver, c("auto", "cholesky", "cd"))
  result$norm <- match.arg(result$norm, c("L1", "L2", "none"))

  result
}


#' Parse advanced SVD parameters from dots
#' @param dots list from list(...)
#' @return Named list with all advanced parameters filled to defaults
#' @keywords internal
.parse_svd_dots <- function(dots) {
  defaults <- list(
    # Advanced regularization
    L21 = 0,
    angular = 0,
    graph_U = NULL,
    graph_V = NULL,
    graph_lambda = 0,
    # Convergence
    convergence = "factor",
    # Cross-validation
    cv_seed = NULL,
    patience = 3,
    k_max = 50,
    # Resources
    threads = 0
  )

  known <- names(defaults)
  unknown <- setdiff(names(dots), known)
  if (length(unknown) > 0) {
    stop("Unknown parameter(s) passed to svd(): ",
         paste0("'", unknown, "'", collapse = ", "),
         ". See ?svd for valid parameters.",
         call. = FALSE)
  }

  result <- defaults
  for (nm in intersect(names(dots), known)) {
    result[[nm]] <- dots[[nm]]
  }

  # Expand paired params
  for (nm in c("L21", "angular", "graph_lambda")) {
    if (length(result[[nm]]) == 1) result[[nm]] <- rep(result[[nm]], 2)
  }

  result$convergence <- match.arg(result$convergence, c("factor", "global"))

  result
}


#' Parse advanced NNLS parameters from dots
#' @param dots list from list(...)
#' @return Named list with all advanced parameters filled to defaults
#' @keywords internal
.parse_nnls_dots <- function(dots) {
  defaults <- list(
    L21 = c(0, 0),
    angular = c(0, 0),
    cd_maxit = 100L,
    cd_tol = 1e-8,
    warm_start = NULL,
    # Target regularization (forwarded to nmf() for target-aware solves)
    target_H = NULL,
    target_lambda = c(0, 0),
    # Distribution tuning (forwarded to nmf() for non-MSE losses)
    dispersion = "per_row",
    theta_init = 0.1,
    theta_max = 5.0,
    theta_min = 0.0,
    nb_size_init = 10.0,
    nb_size_max = 1e6,
    nb_size_min = 0.01,
    gamma_phi_init = 1.0,
    gamma_phi_max = 1e4,
    gamma_phi_min = 1e-6,
    tweedie_power = 1.5,
    irls_max_iter = 5L,
    irls_tol = 1e-4
  )

  known <- names(defaults)
  unknown <- setdiff(names(dots), known)
  if (length(unknown) > 0) {
    stop("Unknown parameter(s) passed to nnls(): ",
         paste0("'", unknown, "'", collapse = ", "),
         ". See ?nnls for valid parameters.",
         call. = FALSE)
  }

  result <- defaults
  for (nm in intersect(names(dots), known)) {
    result[[nm]] <- dots[[nm]]
  }

  for (nm in c("L21", "angular")) {
    if (length(result[[nm]]) == 1) result[[nm]] <- rep(result[[nm]], 2)
  }

  result
}
