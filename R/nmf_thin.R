#' @title Non-negative matrix factorization
#'
#' @description High-performance NMF of the form \eqn{A = wdh} for large dense or sparse matrices. Supports single-rank fitting or cross-validation across multiple ranks. Returns an \code{nmf} object or \code{nmfCrossValidate} data.frame.
#'
#' @details
#' This fast NMF implementation decomposes a matrix \eqn{A} into lower-rank non-negative matrices \eqn{w} and \eqn{h},
#' with columns of \eqn{w} and rows of \eqn{h} scaled to sum to 1 via multiplication by a diagonal, \eqn{d}: \deqn{A = wdh}
#'
#' The scaling diagonal ensures convex L1 regularization, consistent factor scalings regardless of random initialization, and model symmetry in factorizations of symmetric matrices.
#'
#' The factorization model is randomly initialized.  \eqn{w} and \eqn{h} are updated by alternating least squares.
#'
#' RcppML achieves high performance using the Eigen C++ linear algebra library, OpenMP parallelization, a dedicated Rcpp sparse matrix class, and fast sequential coordinate descent non-negative least squares initialized by Cholesky least squares solutions.
#'
#' Sparse optimization is automatically applied if the input matrix \code{A} is a sparse matrix (i.e. \code{Matrix::dgCMatrix}). There are also specialized back-ends for symmetric, rank-1, and rank-2 factorizations.
#'
#' L1 penalization can be used for increasing the sparsity of factors and assisting interpretability. Penalty values should range from 0 to 1, where 1 gives complete sparsity.
#'
#' Set \code{options(RcppML.verbose = TRUE)} to print model tolerances to the console after each iteration.
#'
#' Parallelization is applied with OpenMP using the number of threads in \code{getOption("RcppML.threads")} and set by \code{option(RcppML.threads = 0)}, for example. \code{0} corresponds to all threads, let OpenMP decide.
#'
#' @section Cross-Validation:
#' When \code{k} is a vector, the function performs cross-validation to find the optimal rank:
#' \itemize{
#'   \item A fraction (\code{test_fraction}) of entries are held out for validation
#'   \item Models are fit for each rank in \code{k} using only training data
#'   \item Test loss is computed on held-out entries
#'   \item Returns a data.frame with k, rep, train_mse, test_mse, and best_iter for each rank/replicate
#'   \item Use \code{plot()} on the result to visualize loss vs rank
#' }
#'
#' @section Loss Functions:
#' The \code{loss} parameter controls the objective function:
#' \itemize{
#'   \item \code{"mse"}: Mean Squared Error (default). Standard Frobenius norm minimization.
#'   \item \code{"gp"}: Generalized Poisson. For overdispersed count data (e.g., scRNA-seq).
#'         Use \code{dispersion} to control per-row or global overdispersion estimation.
#'         With \code{dispersion="none"}, equivalent to KL divergence (Poisson model).
#'   \item \code{"nb"}: Negative Binomial. Quadratic variance-mean relationship. Standard for
#'         scRNA-seq data with overdispersion. Size parameter (r) estimated per-row or globally.
#'   \item \code{"gamma"}: Gamma distribution. Variance proportional to mu^2. For positive
#'         continuous data. Dispersion (phi) estimated via MoM Pearson estimator.
#'   \item \code{"inverse_gaussian"}: Inverse Gaussian. Variance proportional to mu^3. For
#'         positive continuous data with heavy right tails.
#'   \item \code{"tweedie"}: Tweedie distribution. Variance proportional to mu^p where p is
#'         set via \code{tweedie_power} (default 1.5). Interpolates between Poisson (p=1),
#'         Gamma (p=2), and Inverse Gaussian (p=3).
#' }
#'
#' For robust fitting (equivalent to Huber or MAE loss), use the \code{robust} parameter
#' instead of a separate loss function. Setting \code{robust = TRUE} applies Huber-weighted
#' IRLS with delta=1.345; \code{robust = "mae"} approximates mean absolute error.
#'
#' Non-MSE losses use Iteratively Reweighted Least Squares (IRLS) which may be slower
#' but provides better fits for count data (GP, NB)
#' and positive continuous data (Gamma, Inverse Gaussian).
#'
#' @section Advanced Parameters (via \code{...}):
#' The following parameters can be passed via \code{...}:
#'
#' \strong{Regularization:}
#' \describe{
#'   \item{\code{L21}}{Group sparsity penalty, single value or \code{c(w, h)} (default \code{c(0,0)})}
#'   \item{\code{angular}}{Angular decorrelation penalty, \code{c(w, h)} (default \code{c(0,0)})}
#'   \item{\code{upper_bound}}{Box constraint on factors, \code{c(w, h)} (default \code{c(0,0)} = no bound)}
#'   \item{\code{graph_W}, \code{graph_H}}{Sparse graph Laplacian matrices for feature/sample regularization}
#'   \item{\code{graph_lambda}}{Graph regularization strength, \code{c(w, h)} (default \code{c(0,0)})}
#'   \item{\code{target_H}}{Target matrix (k x n) for H-side regularization.
#'     Steers H toward a desired structure during fitting. See \code{\link{compute_target}}.}
#'   \item{\code{target_lambda}}{Target regularization strength, single value
#'     or \code{c(w, h)} (default 0). Positive values attract H toward the
#'     target (label enrichment); negative values use PROJ_ADV
#'     eigenvalue-projected adversarial removal to suppress
#'     target-correlated structure (batch removal).
#'     See \code{vignette("guided-nmf")} for details.}
#' }
#'
#' \strong{Distribution Tuning:}
#' \describe{
#'   \item{\code{dispersion}}{Dispersion mode for GP loss: \code{"per_row"}, \code{"per_col"}, \code{"global"}, \code{"none"}}
#'   \item{\code{theta_init}, \code{theta_max}, \code{theta_min}}{GP theta bounds}
#'   \item{\code{nb_size_init}, \code{nb_size_max}, \code{nb_size_min}}{NB dispersion bounds}
#'   \item{\code{gamma_phi_init}, \code{gamma_phi_max}, \code{gamma_phi_min}}{Gamma/IG dispersion bounds}
#'   \item{\code{tweedie_power}}{Tweedie variance power (default 1.5)}
#'   \item{\code{irls_max_iter}, \code{irls_tol}}{IRLS convergence parameters}
#' }
#'
#' \strong{Zero-Inflation:}
#' \describe{
#'   \item{\code{zi_em_iters}}{EM iterations per NMF step (default 1)}
#' }
#'
#' \strong{Solver:}
#' \describe{
#'   \item{\code{solver}}{NNLS solver: \code{"auto"}, \code{"cholesky"}, \code{"cd"}}
#'   \item{\code{cd_tol}}{CD convergence tolerance (default 1e-8)}
#'   \item{\code{cd_maxit}}{CD max iterations (default 100)}
#'   \item{\code{h_init}}{Custom initial H matrix}
#' }
#'
#' \strong{Cross-Validation:}
#' \describe{
#'   \item{\code{cv_seed}}{Separate seed(s) for CV holdout pattern}
#'   \item{\code{patience}}{Early stopping patience (default 5)}
#'   \item{\code{cv_k_range}}{Auto-rank search range (default \code{c(2, 50)})}
#'   \item{\code{track_train_loss}}{Track training loss in CV (default TRUE)}
#' }
#'
#' \strong{Resources & Output:}
#' \describe{
#'   \item{\code{threads}}{OpenMP threads (default 0 = all)}
#'   \item{\code{resource}}{Compute backend: \code{"auto"}, \code{"cpu"}, \code{"gpu"}}
#'   \item{\code{norm}}{Factor normalization: \code{"L1"}, \code{"L2"}, \code{"none"}}
#'   \item{\code{sort_model}}{Sort factors by diagonal (default TRUE)}
#' }
#'
#' \strong{Streaming:}
#' \describe{
#'   \item{\code{streaming}}{SPZ streaming mode: \code{"auto"}, \code{TRUE}, \code{FALSE}}
#'   \item{\code{panel_cols}}{Panel size for streaming (default 0 = auto)}
#'   \item{\code{dispatch}}{StreamPress dispatch override}
#' }
#'
#' \strong{Callbacks:}
#' \describe{
#'   \item{\code{on_iteration}}{Per-iteration callback function}
#'   \item{\code{profile}}{Enable timing profiling (default FALSE)}
#' }
#'
#' @section Compute Resources (CPU vs GPU):
#' By default (\code{resource = "auto"}), RcppML auto-detects available hardware.
#' All features are fully supported on CPU (the default backend).
#' GPU acceleration (when compiled with CUDA support) accelerates sparse and dense NMF.
#' GPU is experimental and falls back to CPU automatically if unavailable.
#'
#' @section Methods:
#' S4 methods available for the \code{nmf} class:
#' * \code{predict}: project an NMF model (or partial model) onto new samples
#' * \code{evaluate}: calculate mean squared error loss of an NMF model
#' * \code{summary}: \code{data.frame} giving \code{fractional}, \code{total}, or \code{mean} representation of factors in samples or features grouped by some criteria
#' * \code{align}: find an ordering of factors in one \code{nmf} model that best matches those in another \code{nmf} model
#' * \code{prod}: compute the dense approximation of input data
#' * \code{sparsity}: compute the sparsity of each factor in \eqn{w} and \eqn{h}
#' * \code{subset}: subset, reorder, select, or extract factors (same as `[`)
#' * generics such as \code{dim}, \code{dimnames}, \code{t}, \code{show}, \code{head}
#'
#' @param data dense or sparse matrix of features in rows and samples in columns. Prefer \code{matrix} or \code{Matrix::dgCMatrix}, respectively.
#'   Also accepts a file path (character string) which will be auto-loaded based on extension.
#' @param k rank (integer), vector of ranks for cross-validation, or "auto" for automatic rank determination.
#' @param tol tolerance of the fit (default 1e-4)
#' @param maxit maximum number of fitting iterations (default 100)
#' @param L1 LASSO penalties in the range (0, 1], single value or array of length two for \code{c(w, h)}
#' @param L2 Ridge penalties greater than zero, single value or array of length two for \code{c(w, h)}
#' @param seed initialization control. Accepts:
#'   \code{NULL} for random init, an integer for reproducible random init,
#'   a matrix (m x k) for custom W initialization, a list of matrices for multi-init,
#'   or a string: \code{"lanczos"}, \code{"irlba"}, \code{"randomized"}, or \code{"svd"}
#'   (auto-select) for SVD-based initialization.
#' @param mask missing data mask. Accepts:
#'   \code{NULL} (no masking), \code{"zeros"} (mask zeros), \code{"NA"} (mask NAs),
#'   a dgCMatrix/matrix (custom mask), or \code{list("zeros", <matrix>)} to mask
#'   zeros and use a custom mask simultaneously.
#' @param loss loss function: \code{"mse"} (default), \code{"gp"},
#'   \code{"nb"}, \code{"gamma"}, \code{"inverse_gaussian"}, or \code{"tweedie"}.
#'   For robust loss (Huber/MAE), use the \code{robust} parameter instead.
#' @param nonneg logical vector of length 2 for \code{c(w, h)} specifying non-negativity constraints (default \code{c(TRUE, TRUE)}).
#' @param test_fraction fraction of entries to hold out for cross-validation (default 0 = disabled).
#' @param verbose print progress information during fitting (default FALSE)
#' @param projective if \code{TRUE}, use projective NMF (default \code{FALSE}).
#' @param symmetric if \code{TRUE}, use symmetric NMF where H = W^T (default \code{FALSE}).
#' @param zi zero-inflation mode: \code{"none"} (default), \code{"row"}, or \code{"col"}.
#'   Requires \code{loss="gp"} or \code{loss="nb"}.
#' @param robust robustness control. \code{FALSE} (default), \code{TRUE} (Huber delta=1.345),
#'   \code{"mae"} (near-MAE via very small Huber delta), or a positive numeric Huber delta.
#'   Huber loss is quadratic for small residuals and linear for large ones, controlled by delta.
#'   \code{TRUE} (delta=1.345) provides moderate outlier robustness.
#'   A large delta (e.g. 100) approaches standard MSE; a small delta (e.g. 0.01) approaches MAE.
#'   \code{"mae"} is shorthand for delta=1e-4 (effectively L1 loss).
#' @param ... advanced parameters. See \strong{Advanced Parameters} section.
#' @return
#' When \code{k} is a single integer: an S4 object of class \code{nmf} with slots:
#' \describe{
#'   \item{\code{w}}{feature factor matrix, \code{m x k}}
#'   \item{\code{d}}{scaling diagonal vector of length \code{k} (descending order after sorting)}
#'   \item{\code{h}}{sample factor matrix, \code{k x n}}
#'   \item{\code{misc}}{list containing \code{tol} (final tolerance), \code{iter} (iteration count),
#'     \code{loss} (final loss value), \code{loss_type} (loss function used), and \code{runtime} (seconds).}
#' }
#' When \code{k} is a vector: a \code{data.frame} of class \code{nmfCrossValidate} with columns
#' \code{k}, \code{rep}, \code{train_loss}, \code{test_loss}, and \code{best_iter}.
#' @importFrom methods is
#' @importFrom stats runif sd setNames var
#' @references
#'
#' DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance non-negative matrix factorization for large single-cell data." BioRXiv.
#'
#' @author Zach DeBruine
#'
#' @export
#' @seealso \code{\link{predict}}, \code{\link{mse}}, \code{\link{nnls}},
#'   \code{\link{align}}, \code{\link{summary,nmf-method}}
#' @md
#' @examples
#' \donttest{
#' # basic NMF
#' model <- nmf(Matrix::rsparsematrix(1000, 100, 0.1), k = 10)
#'
#' # cross-validation to find optimal rank
#' sim <- simulateNMF(200, 80, k = 5, noise = 3.0, seed = 42)
#' cv <- nmf(sim$A, k = 2:10, test_fraction = 0.05, cv_seed = 1:3,
#'           tol = 1e-5, maxit = 200)
#' plot(cv)
#' optimal_k <- cv$k[which.min(cv$test_mse)]
#' 
#' # fit final model with optimal rank
#' model <- nmf(sim$A, optimal_k)
#' }
nmf <- function(data, k, tol = 1e-4, maxit = 100, L1 = c(0, 0), L2 = c(0, 0),
                seed = NULL, mask = NULL,
                loss = c("mse", "gp", "nb", "gamma", "inverse_gaussian", "tweedie"),
                nonneg = c(TRUE, TRUE),
                test_fraction = 0,
                verbose = FALSE,
                projective = FALSE,
                symmetric = FALSE,
                zi = c("none", "row", "col"),
                robust = FALSE,
                ...) {
  
  start_time <- Sys.time()
  
  # --- Parse advanced parameters from ... ---
  dots <- .parse_nmf_dots(list(...))
  
  # Extract to local variables for downstream code
  L21           <- dots$L21
  angular       <- dots$angular
  upper_bound   <- dots$upper_bound
  graph_W       <- dots$graph_W
  graph_H       <- dots$graph_H
  graph_lambda  <- dots$graph_lambda
  target_H      <- dots$target_H
  target_lambda <- dots$target_lambda
  dispersion    <- dots$dispersion
  theta_init    <- dots$theta_init
  theta_max     <- dots$theta_max
  theta_min     <- dots$theta_min
  nb_size_init  <- dots$nb_size_init
  nb_size_max   <- dots$nb_size_max
  nb_size_min   <- dots$nb_size_min
  gamma_phi_init <- dots$gamma_phi_init
  gamma_phi_max <- dots$gamma_phi_max
  gamma_phi_min <- dots$gamma_phi_min
  tweedie_power <- dots$tweedie_power
  irls_max_iter <- dots$irls_max_iter
  irls_tol      <- dots$irls_tol
  zi_em_iters   <- dots$zi_em_iters
  solver        <- dots$solver
  cd_tol        <- dots$cd_tol
  cd_maxit      <- dots$cd_maxit
  h_init        <- dots$h_init
  w_init_arg    <- dots$w_init
  cv_seed       <- dots$cv_seed
  patience      <- dots$patience
  cv_k_range    <- dots$cv_k_range
  track_train_loss <- dots$track_train_loss
  threads       <- dots$threads
  resource      <- dots$resource
  norm          <- dots$norm
  sort_model    <- dots$sort_model
  streaming     <- dots$streaming
  panel_cols    <- dots$panel_cols
  dispatch      <- dots$dispatch
  on_iteration  <- dots$on_iteration
  profile       <- dots$profile
  
  # --- Multi-modal list input: delegate to factor_net ---
  if (is.list(data) && !inherits(data, "Matrix") && !inherits(data, "gpu_sparse_matrix")) {
    if (length(data) < 2)
      stop("Multi-modal NMF requires a list of 2+ matrices with the same number of columns")
    nms <- names(data)
    if (is.null(nms) || any(nms == "") || anyDuplicated(nms))
      stop("Multi-modal NMF requires a named list with unique names")
    # All matrices must have the same number of columns (shared samples)
    ncols <- vapply(data, ncol, integer(1))
    if (length(unique(ncols)) != 1)
      stop("All matrices in multi-modal NMF must have the same number of columns (samples)")
    
    # Build factor_shared graph
    inputs <- lapply(nms, function(nm) factor_input(data[[nm]], nm))
    shared <- do.call(factor_shared, inputs)
    layer <- nmf_layer(shared, k = k,
                       W = W(L1 = L1[1], L2 = L2[1], L21 = L21[1],
                             angular = angular[1], nonneg = nonneg[1],
                             upper_bound = upper_bound[1]),
                       H = H(L1 = L1[2], L2 = L2[2], L21 = L21[2],
                             angular = angular[2], nonneg = nonneg[2],
                             upper_bound = upper_bound[2]))
    cfg <- factor_config(maxit = maxit, tol = tol, seed = seed,
                         verbose = verbose)
    net <- factor_net(inputs = inputs, output = layer, config = cfg)
    return(fit(net))
  }
  
  # --- Resource override: param > env var > options > "auto" ---
  if (resource == "auto") {
    env_res <- Sys.getenv("RCPPML_RESOURCE", "auto")
    if (env_res != "auto") {
      resource <- tolower(env_res)
    } else {
      gpu_opt <- getOption("RcppML.gpu", "auto")
      if (isTRUE(gpu_opt)) {
        resource <- "gpu"
      } else if (identical(gpu_opt, FALSE)) {
        resource <- "cpu"
      }
      # If gpu_opt is "auto" or anything else, resource stays "auto"
    }
  }
  if (!resource %in% c("auto", "cpu", "gpu")) {
    stop("'resource' must be \"auto\", \"cpu\", or \"gpu\"")
  }
  
  # Resolve "auto" → check if GPU is actually available
  use_gpu <- FALSE
  if (resource == "gpu") {
    if (gpu_available()) {
      use_gpu <- TRUE
    } else {
      warning("GPU requested but not available, falling back to CPU")
    }
  } else if (resource == "auto" && gpu_available()) {
    use_gpu <- TRUE
  }
  
  # --- Match string arguments ---
  loss <- match.arg(loss)
  dispersion_mode <- switch(dispersion, none = 0L, global = 1L, per_row = 2L, per_col = 3L)
  zi <- match.arg(zi)
  zi_mode <- switch(zi, none = 0L, row = 1L, col = 2L)

  # Parse robust parameter: FALSE=0, TRUE=1.345, "mae"=1e-4, numeric=value
  if (is.logical(robust)) {
    robust_delta <- if (isTRUE(robust)) 1.345 else 0.0
  } else if (is.character(robust) && tolower(robust) == "mae") {
    robust_delta <- 1e-4
  } else if (is.numeric(robust)) {
    robust_delta <- as.double(robust)
  } else {
    stop("'robust' must be FALSE, TRUE, 'mae', or a positive numeric Huber delta.", call. = FALSE)
  }

  # huber_delta: only relevant for legacy C++ paths, always 0.0 (robust handled by robust_delta)
  huber_delta <- 0.0

  # ZIGP/ZINB requires GP or NB loss
  if (zi != "none") {
    if (!loss %in% c("gp", "nb"))
      stop("zi != 'none' requires loss='gp' or loss='nb'.", call. = FALSE)
  }

  # Auto solver selection
  if (solver == "auto") {
    needs_irls <- (loss != "mse") || (robust_delta > 0)
    if (needs_irls) {
      solver <- "cd"
    } else if (use_gpu) {
      solver <- if (max(k) <= 32) "cd" else "cholesky"
    } else {
      if (max(k) < 32 && all(L1 == 0)) {
        solver <- "cholesky"
      } else {
        solver <- "cd"
      }
    }
  }
  if (solver == "cholesky" && loss != "mse" && robust_delta == 0) {
    stop("solver='cholesky' is not supported with non-MSE distributions (got '", loss, "'). ",
         "Use solver='cd' for IRLS-based distributions.",
         call. = FALSE)
  }
  if (solver == "cholesky" && robust_delta > 0) {
    stop("solver='cholesky' is not supported with robust IRLS (robust_delta > 0). ",
         "Use solver='cd' for robust estimation.", call. = FALSE)
  }

  solver_mode <- switch(solver, cd = 0L, cholesky = 1L, 0L)
  
  # Derive init_mode from seed: string seeds map to SVD init methods
  init_mode <- 0L  # random default
  if (is.character(seed) && length(seed) == 1) {
    init_mode <- switch(seed,
      random   = 0L, 
      lanczos  = 1L,
      irlba    = 2L,
      randomized = 1L,
      svd      = 1L,
      stop("Unknown seed string '", seed, "'. Valid: NULL, integer, matrix, ",
           "'lanczos', 'irlba', 'randomized', 'svd'", call. = FALSE)
    )
    seed <- NULL  # clear so downstream seed handling treats as NULL
  }
  

  
  # --- GPU sparse matrix input: zero-copy GPU NMF ---
  # sp_read_gpu() returns a gpu_sparse_matrix with data already on device.
  # Route directly to the zero-copy GPU NMF path to avoid re-uploading.
  is_gpu_sparse <- inherits(data, "gpu_sparse_matrix")
  if (is_gpu_sparse) {
    if (!gpu_available()) {
      stop("gpu_sparse_matrix requires GPU. Use sp_read() + nmf() for CPU.",
           call. = FALSE)
    }
    use_gpu <- TRUE
    # Extract dimensions for downstream validation
    n_features <- data$m
    n_samples <- data$n
  }
  
  # --- StreamPress auto-dispatch ---
  # If data is a .spz file path, auto-detect the optimal execution mode
  # based on available RAM/VRAM and estimated data size.
  if (is.character(data) && length(data) == 1 &&
      grepl("\\.spz$", data, ignore.case = TRUE)) {
    if (is.null(dispatch) || identical(dispatch, "auto")) {
      mode_info <- .st_dispatch(data, k, resource = resource)
      if (verbose) message("StreamPress auto-dispatch: ", mode_info$mode)
      resource <- mode_info$resource
      streaming <- mode_info$streaming
    } else {
      message(
        "WARNING: `dispatch` is set manually to '", dispatch, "'.\n",
        "  Auto-dispatch ensures sufficient RAM is available before loading.\n",
        "  Manual dispatch may cause out-of-memory errors or crashes.\n",
        "  Remove `dispatch=` to restore safe automatic mode."
      )
    }
  }

  # --- Streaming SPZ shortcut ---
  # If data is a .spz file path AND streaming is enabled,
  # use streaming NMF directly (no decompression into memory).
  # This requires a v2 .spz file with include_transpose=TRUE.
  # Supports standard NMF, NA masking, and cross-validation (CV).
  # Streaming is enabled when: streaming=TRUE, or streaming="auto" with SPZ input,
  # or the global option RcppML.streaming is TRUE.
  use_streaming <- isTRUE(streaming) ||
    (identical(streaming, "auto") && is.character(data) && length(data) == 1 &&
     grepl("\\.spz$", data)) ||
    isTRUE(getOption("RcppML.streaming", FALSE))
  if (is.character(data) && length(data) == 1 && grepl("\\.spz$", data) &&
      use_streaming) {
    if (!file.exists(data)) stop("SPZ file not found: ", data)

    # Validate mask for streaming
    spz_mask <- NULL  # dgCMatrix or NULL
    spz_mask_zeros <- FALSE
    if (!is.null(mask)) {
      if (is.character(mask)) {
        if (mask == "NA") {
          stop("Streaming SPZ NMF does not support mask = \"NA\". ",
               "NA detection requires the full matrix in memory. ",
               "Provide an explicit mask matrix or disable streaming.")
        } else if (mask == "zeros") {
          spz_mask_zeros <- TRUE
        } else {
          stop("'mask' must be a matrix, 'zeros', or NULL for streaming SPZ NMF")
        }
      } else if (is.list(mask) && length(mask) == 2 && identical(mask[[1]], "zeros")) {
        spz_mask_zeros <- TRUE
        spz_mask <- .to_dgCMatrix(as(mask[[2]], "dMatrix"))
      } else {
        spz_mask <- .to_dgCMatrix(as(mask, "dMatrix"))
      }
    }

    # NB streaming: fixed dispersion warning (§2.2.1)
    if (loss == "nb") {
      warning(
        "Streaming NB NMF uses fixed dispersion throughout all iterations.\n",
        "  The in-memory path re-estimates dispersion per iteration, but streaming cannot.\n",
        "  Supply a pre-estimated nb_size_init if your data has strong overdispersion.",
        call. = FALSE
      )
    }

    if (length(nonneg) == 1) nonneg <- rep(nonneg, 2)
    seed_int <- if (is.null(seed)) sample.int(.Machine$integer.max, 1) else as.integer(seed)

    # Convert graph matrices to dgCMatrix (Rcpp expects S4)
    if (!is.null(graph_W) && !is(graph_W, "dgCMatrix")) {
      graph_W <- .to_dgCMatrix(as(graph_W, "dMatrix"))
    }
    if (!is.null(graph_H) && !is(graph_H, "dgCMatrix")) {
      graph_H <- .to_dgCMatrix(as(graph_H, "dMatrix"))
    }

    # Prepare graph lambda vector
    graph_lambda_vec <- c(
      if (!is.null(graph_W)) graph_lambda[1] else 0,
      if (!is.null(graph_H)) {
        if (length(graph_lambda) > 1) graph_lambda[2] else graph_lambda[1]
      } else 0
    )

    # Handle k
    k_vec <- as.integer(k)

    # CV seeds
    cv_seeds_spz <- if (!is.null(cv_seed)) as.integer(cv_seed) else seed_int

    # --- Multi-rank CV sweep ---
    if (length(k_vec) > 1) {
      if (test_fraction <= 0) {
        test_fraction <- 0.1
        if (verbose) message("Setting test_fraction = 0.1 for cross-validation across ranks")
      }
      reps <- length(cv_seeds_spz)
      results <- vector("list", length(k_vec) * reps)
      idx <- 1L
      for (rep_idx in seq_len(reps)) {
        rep_cv_seed <- cv_seeds_spz[rep_idx]
        for (rank in k_vec) {
          init_seed <- as.integer((rep_cv_seed + rank) %% .Machine$integer.max)
          result_i <- Rcpp_nmf_streaming_spz(
            path = data, k = rank, tol = tol, maxit = as.integer(maxit),
            L1 = as.double(L1), L2 = as.double(L2), L21 = as.double(L21),
            angular = as.double(angular), upper_bound = as.double(upper_bound),
            graph_lambda = as.double(graph_lambda_vec),
            nonneg_w = nonneg[1], nonneg_h = nonneg[2],
            cd_maxit = as.integer(cd_maxit), cd_tol = cd_tol,
            threads = as.integer(threads), seed = init_seed,
            verbose = FALSE,
            loss_str = loss, huber_delta = huber_delta,
            holdout_fraction = test_fraction,
            mask_zeros = spz_mask_zeros,
            cv_seed = as.integer(rep_cv_seed),
            norm_str = norm,
            mask_sexp = spz_mask,
            graph_W_sexp = graph_W, graph_H_sexp = graph_H,
            solver_mode = solver_mode,
            init_mode = init_mode,
            projective = projective,
            symmetric = symmetric
          )
          results[[idx]] <- data.frame(
            rep = rep_idx, k = rank,
            train_mse = result_i$loss,
            test_mse = result_i$test_loss,
            best_iter = result_i$iter,
            total_iter = result_i$iter
          )
          idx <- idx + 1L
        }
      }
      df <- do.call(rbind, results)
      class(df) <- c("nmfCrossValidate", "data.frame")
      df$rep <- as.factor(df$rep)
      return(df)
    }

    # --- Single-rank: standard NMF, optionally with CV if test_fraction > 0 ---
    k_val <- k_vec[1]

    result <- Rcpp_nmf_streaming_spz(
      path = data,
      k = k_val,
      tol = tol,
      maxit = as.integer(maxit),
      L1 = as.double(L1),
      L2 = as.double(L2),
      L21 = as.double(L21),
      angular = as.double(angular),
      upper_bound = as.double(upper_bound),
      graph_lambda = as.double(graph_lambda_vec),
      nonneg_w = nonneg[1],
      nonneg_h = nonneg[2],
      cd_maxit = as.integer(cd_maxit),
      cd_tol = cd_tol,
      threads = as.integer(threads),
      seed = seed_int,
      verbose = verbose,
      loss_str = loss,
      huber_delta = huber_delta,
      holdout_fraction = test_fraction,
      mask_zeros = spz_mask_zeros,
      cv_seed = as.integer(cv_seeds_spz[1]),
      norm_str = norm,
      mask_sexp = spz_mask,
      graph_W_sexp = graph_W,
      graph_H_sexp = graph_H,
      solver_mode = solver_mode,
      init_mode = init_mode,
      projective = projective,
      symmetric = symmetric,
      enable_profiling = profile
    )
    
    k_final <- ncol(result$w)
    colnames(result$w) <- rownames(result$h) <- paste0("nmf", seq_len(k_final))
    
    misc <- list(
      tol = result$loss,
      iter = result$iter,
      loss_type = loss,
      loss = result$loss,
      runtime = difftime(Sys.time(), start_time, units = "secs"),
      streaming = TRUE,
      spz_file = data,
      solver_mode = solver_mode
    )
    if (!is.null(result$train_loss_history) && length(result$train_loss_history) > 0)
      misc$loss_history <- result$train_loss_history
    if (test_fraction > 0) {
      misc$test_loss <- result$test_loss
      if (!is.null(result$test_loss_history) && length(result$test_loss_history) > 0)
        misc$test_loss_history <- result$test_loss_history
    }
    if (!is.null(result$profile)) {
      misc$profile <- result$profile
    }
    
    return(
      new("nmf", w = result$w, d = result$d, h = result$h, misc = misc))
  }
  
  # --- Load file / coerce matrix type / detect NAs via shared helper ---
  if (is_gpu_sparse) {
    # gpu_sparse_matrix: data stays on device, skip host validation
    is_sparse <- TRUE
    has_na <- FALSE
  } else {
    vd <- validate_data(data)
    data     <- vd$data
    is_sparse <- vd$is_sparse
    has_na   <- vd$has_na
  }
  
  # --- Validate penalties ---
  vp <- validate_all_penalties(L1, L2, L21, angular, graph_lambda, upper_bound)
  L1 <- vp$L1; L2 <- vp$L2; L21 <- vp$L21; angular <- vp$angular
  graph_lambda <- vp$graph_lambda; upper_bound <- vp$upper_bound
  
  # --- Validate target_lambda: scalar → c(0, scalar) since target_H is H-side ---
  if (length(target_lambda) == 1)
    target_lambda <- c(0, target_lambda)
  
  # --- Validate simple params ---
  vs <- validate_simple_params(sort_model, nonneg)
  sort_model <- vs$sort_model; nonneg <- vs$nonneg
  
  # --- Validate projective / symmetric ---
  projective <- as.logical(projective)[1]
  symmetric  <- as.logical(symmetric)[1]
  if (is.na(projective)) projective <- FALSE
  if (is.na(symmetric))  symmetric  <- FALSE
  if (projective && symmetric) {
    stop("'projective' and 'symmetric' cannot both be TRUE", call. = FALSE)
  }
  if (symmetric) {
    dims <- dim(data)
    if (dims[1] != dims[2]) {
      warning("symmetric NMF is intended for square (symmetric) matrices; ",
              "data has dimensions ", dims[1], " x ", dims[2], call. = FALSE)
    }
  }
  
  # --- Validate CV params ---
  vc <- validate_cv_params(test_fraction, patience)
  test_fraction <- vc$test_fraction; patience <- vc$patience
  
  # --- Validate graphs ---
  vg <- validate_graphs(graph_W, graph_H, dim(data), graph_lambda)
  graph_W <- vg$graph_W; graph_H <- vg$graph_H

  # Build graph_lambda_vec used by GPU dispatch and other paths
  graph_lambda_vec <- c(
    if (!is.null(graph_W)) graph_lambda[1] else 0,
    if (!is.null(graph_H)) {
      if (length(graph_lambda) > 1) graph_lambda[2] else graph_lambda[1]
    } else 0
  )

  # --- Handle NA values ---
  if (has_na) {
    if (is_sparse) {
      na_count <- sum(is.na(data@x))
      data@x[is.na(data@x)] <- 0
    } else {
      na_count <- sum(is.na(data))
      data[is.na(data)] <- 0
    }
    if (is.null(mask)) mask <- "NA"
  }
  
  # --- Handle mask → mask_zeros flag ---
  vm <- validate_mask(mask, has_na)
  mask_zeros <- vm$mask_zeros
  mask <- vm$mask_matrix
  
  # --- Handle k ---
  k_auto <- FALSE
  k_auto_min <- cv_k_range[1]
  k_auto_max <- cv_k_range[2]
  is_cv <- FALSE
  
  if (is.character(k) && k == "auto") {
    k_auto <- TRUE
    k_vec <- as.integer(cv_k_range[1])
    if (test_fraction == 0) test_fraction <- 0.1
    is_cv <- TRUE
  } else {
    k_vec <- as.integer(k)
    if (length(k_vec) > 1) is_cv <- TRUE
  }
  
  if (test_fraction > 0) is_cv <- TRUE
  
  n_features <- nrow(data)
  
  # --- Handle seed → w_init + seed_int ---
  # For random init (init_mode == 0L): use R's RNG for initialization (matching legacy behavior)
  # For SVD init (init_mode == 1L or 2L): pass seed_int to C++, let C++ compute SVD
  w_init_mat <- NULL
  w_init_list <- NULL
  seed_int <- 0L
  
  # Determine if we should create w_init_mat in R (only for random init)
  use_r_init <- (init_mode == 0L)
  
  if (is.null(seed) || (is.character(seed) && seed == "random")) {
    # Random init using R's current RNG state (matching legacy behavior)
    if (use_r_init) {
      # Do NOT consume extra RNG draws — derive seed_int from W_init values
      w_init_mat <- matrix(runif(n_features * k_vec[1]), n_features, k_vec[1])
      seed_int <- as.integer(abs(sum(w_init_mat[1:min(10, length(w_init_mat))]) * 1e8) %% .Machine$integer.max)
    } else {
      # SVD init: generate seed_int without creating w_init_mat
      seed_int <- as.integer(abs(runif(1) * .Machine$integer.max))
    }
    
  } else if (is.list(seed)) {
    # List of custom W matrices for multi-init
    w_init_list <- lapply(seed, function(s) {
      if (!is.matrix(s)) stop("Each element of seed list must be a matrix")
      actual_k <- if (nrow(s) == n_features) ncol(s) else nrow(s)
      if (actual_k != k_vec[1]) {
        stop(sprintf("Rank mismatch: k=%d specified but custom initialization has rank %d.", k_vec[1], actual_k))
      }
      if (nrow(s) == n_features) s
      else if (ncol(s) == n_features) t(s)
      else stop("Custom init matrix dimensions incompatible with data")
    })
    seed_int <- abs(as.integer(sum(seed[[1]] * 1e6) %% .Machine$integer.max))
    
  } else if (is.matrix(seed)) {
    # Single custom W initialization
    actual_k <- if (nrow(seed) == n_features) ncol(seed) else nrow(seed)
    if (actual_k != k_vec[1]) {
      stop(sprintf("Rank mismatch: k=%d specified but custom initialization has rank %d.", k_vec[1], actual_k))
    }
    if (nrow(seed) == n_features) {
      w_init_mat <- seed
    } else if (ncol(seed) == n_features) {
      w_init_mat <- t(seed)
    } else {
      stop("Custom init matrix dimensions incompatible with data")
    }
    seed_int <- abs(as.integer(sum(seed * 1e6) %% .Machine$integer.max))
    
  } else if (is.numeric(seed) && length(seed) > 1) {
    # Vector of seeds for multi-init
    if (use_r_init) {
      w_init_list <- lapply(seed, function(s) {
        set.seed(as.integer(s))
        matrix(runif(n_features * k_vec[1]), n_features, k_vec[1])
      })
    } else {
      # SVD init with multiple seeds: not supported (SVD is deterministic from seed)
      # Just use the first seed
      seed_int <- as.integer(seed[1])
      if (length(seed) > 1) {
        warning("Multiple seeds provided with SVD initialization - only the first seed will be used")
      }
    }
    seed_int <- as.integer(seed[1])
    
  } else if (is.numeric(seed)) {
    # Single numeric seed
    seed_int <- as.integer(seed)
    if (use_r_init) {
      set.seed(seed_int)
      w_init_mat <- matrix(runif(n_features * k_vec[1]), n_features, k_vec[1])
    }
    # else: SVD init, just pass seed_int to C++
  }
  
  # --- Handle cv_seed ---
  cv_seeds <- if (!is.null(cv_seed)) as.integer(cv_seed) else integer(0)
  
  # --- Multi-rank CV: ensure holdout > 0 ---
  if (length(k_vec) > 1 && test_fraction == 0) {
    test_fraction <- 0.1
    if (verbose) message("Setting test_fraction = 0.1 for cross-validation across ranks")
  }
  
  # --- CV seeds for multi-rank: derive from seed if not provided ---
  if (length(k_vec) > 1 && length(cv_seeds) == 0) {
    cv_seeds <- as.integer(seed_int)
  }
  
  # --- Handle explicit w_init from ... ---
  if (!is.null(w_init_arg)) {
    if (!is.matrix(w_init_arg)) stop("w_init must be a matrix")
    if (nrow(w_init_arg) == n_features && ncol(w_init_arg) == k_vec[1]) {
      w_init_mat <- w_init_arg
    } else if (ncol(w_init_arg) == n_features && nrow(w_init_arg) == k_vec[1]) {
      w_init_mat <- t(w_init_arg)
    } else {
      stop("w_init dimensions incompatible with data and k")
    }
    if (seed_int == 0L)
      seed_int <- abs(as.integer(sum(w_init_mat[1:min(10, length(w_init_mat))]) * 1e8) %% .Machine$integer.max)
  }

  # --- Multiple initializations: run each, pick best ---
  if (!is.null(w_init_list) && length(w_init_list) > 1) {
    if (test_fraction > 0) {
      stop("Multiple initializations are not compatible with cross-validation. Use a single seed or matrix.")
    }
    
    best_loss <- Inf
    best_result <- NULL
    best_idx <- 1L
    all_losses <- numeric(length(w_init_list))
    
    for (i in seq_along(w_init_list)) {
      result_i <- Rcpp_nmf_full(
        A_sexp = data, k_vec = k_vec, k_auto = FALSE,
        k_auto_min = 2L, k_auto_max = 50L,
        L1 = as.double(L1), L2 = as.double(L2),
        L21 = as.double(L21), angular = as.double(angular),
        upper_bound = as.double(upper_bound),
        graph_lambda = as.double(graph_lambda),
        tol = tol, maxit = as.integer(maxit),
        cd_maxit = as.integer(cd_maxit), cd_tol = cd_tol,
        threads = as.integer(threads),
        seed = as.integer(seed_int + i - 1L),
        nonneg_w = nonneg[1], nonneg_h = nonneg[2],
        sort_model = sort_model, mask_zeros = mask_zeros,
        track_loss_history = TRUE, track_train_loss = FALSE,
        loss_str = loss, huber_delta = huber_delta,
        holdout_fraction = 0, cv_seeds = integer(0), cv_patience = 0L,
        resource_override = resource,
        norm_str = norm,
        graph_W_sexp = graph_W, graph_H_sexp = graph_H,
        w_init_sexp = w_init_list[[i]],
        h_init_sexp = h_init,
        verbose = FALSE,
        mask_sexp = if (inherits(mask, "sparseMatrix")) mask else NULL,
        projective = projective,
        symmetric_nmf = symmetric,
        solver_mode = solver_mode,
        init_mode = init_mode,
        irls_max_iter = as.integer(irls_max_iter),

        irls_tol = irls_tol,
        gp_dispersion_mode = dispersion_mode,
        gp_theta_init = theta_init,
        gp_theta_max = theta_max,
        zi_mode = zi_mode,
        zi_em_iters = as.integer(zi_em_iters),
        gp_theta_min = theta_min,
        nb_size_init = nb_size_init,
        nb_size_max = nb_size_max,
        nb_size_min = nb_size_min,
        gamma_phi_init = gamma_phi_init,
        gamma_phi_max = gamma_phi_max,
        gamma_phi_min = gamma_phi_min,
        robust_delta = robust_delta,
        tweedie_power = tweedie_power,
        target_H_sexp = target_H,
        target_lambda = target_lambda,
        enable_profiling = profile
      )
      all_losses[i] <- result_i$loss
      if (result_i$loss < best_loss) {
        best_loss <- result_i$loss
        best_result <- result_i
        best_idx <- i
      }
    }
    
    result <- best_result
    
    # Build S4 result
    k_final <- ncol(result$w)
    colnames(result$w) <- rownames(result$h) <- paste0("nmf", 1:k_final)
    if (!is.null(rownames(data))) rownames(result$w) <- rownames(data)
    if (!is.null(colnames(data))) colnames(result$h) <- colnames(data)
    
    misc <- list(
      tol = result$tol,
      iter = result$iter,
      loss_type = loss,
      loss = result$loss,
      runtime = difftime(Sys.time(), start_time, units = "secs"),
      w_init = w_init_list[[best_idx]],
      all_inits = data.frame(
        init = seq_along(all_losses),
        loss = all_losses,
        selected = seq_along(all_losses) == best_idx
      )
    )
    
    return(
      new("nmf", w = result$w, d = result$d, h = result$h, misc = misc))
  }
  
  # --- k = "auto": automatic rank search using C++ rank search engine ---
  if (k_auto) {
    cv_seed_auto <- if (length(cv_seeds) > 0) cv_seeds[1] else seed_int
    rank_result <- .Call(`_RcppML_Rcpp_nmf_rank_cv`,
      data, as.integer(k_auto_min), as.integer(k_auto_max), 1L,
      test_fraction, as.integer(cv_seed_auto),
      tol, as.integer(maxit), verbose,
      as.double(L1), as.double(L2), as.integer(threads))
    
    # Fit final model at optimal rank
    k_vec <- as.integer(rank_result$k_optimal)
    
    # Generate fresh W_init for optimal rank
    if (is.null(w_init_mat) || ncol(w_init_mat) != k_vec[1]) {
      w_init_mat <- matrix(runif(n_features * k_vec[1]), n_features, k_vec[1])
    }
    
    result <- Rcpp_nmf_full(
      A_sexp = data, k_vec = k_vec, k_auto = FALSE,
      k_auto_min = as.integer(k_auto_min), k_auto_max = as.integer(k_auto_max),
      L1 = as.double(L1), L2 = as.double(L2),
      L21 = as.double(L21), angular = as.double(angular),
      upper_bound = as.double(upper_bound),
      graph_lambda = as.double(graph_lambda),
      tol = tol, maxit = as.integer(maxit),
      cd_maxit = as.integer(cd_maxit), cd_tol = cd_tol,
      threads = as.integer(threads),
      seed = as.integer(seed_int),
      nonneg_w = nonneg[1], nonneg_h = nonneg[2],
      sort_model = sort_model, mask_zeros = mask_zeros,
      track_loss_history = TRUE, track_train_loss = track_train_loss,
      loss_str = loss, huber_delta = huber_delta,
      holdout_fraction = 0, cv_seeds = integer(0), cv_patience = 0L,
      resource_override = resource,
      norm_str = norm,
      graph_W_sexp = graph_W, graph_H_sexp = graph_H,
      w_init_sexp = w_init_mat,
      h_init_sexp = h_init,
      verbose = verbose,
      mask_sexp = if (!is.null(mask) && inherits(mask, "sparseMatrix")) mask else NULL,
      projective = projective,
      symmetric_nmf = symmetric,
      solver_mode = solver_mode,
      init_mode = init_mode,
      irls_max_iter = as.integer(irls_max_iter),
      irls_tol = irls_tol,
      gp_dispersion_mode = dispersion_mode,
      gp_theta_init = theta_init,
      gp_theta_max = theta_max,
      zi_mode = zi_mode,
      zi_em_iters = as.integer(zi_em_iters),
      gp_theta_min = theta_min,
      nb_size_init = nb_size_init,
      nb_size_max = nb_size_max,
      nb_size_min = nb_size_min,
      gamma_phi_init = gamma_phi_init,
      gamma_phi_max = gamma_phi_max,
      gamma_phi_min = gamma_phi_min,
      robust_delta = robust_delta,
      tweedie_power = tweedie_power,
      target_H_sexp = target_H,
      target_lambda = target_lambda,
      enable_profiling = profile
    )
    
    # Build S4 result with rank_search metadata
    k_final <- ncol(result$w)
    colnames(result$w) <- rownames(result$h) <- paste0("nmf", 1:k_final)
    if (!is.null(rownames(data))) rownames(result$w) <- rownames(data)
    if (!is.null(colnames(data))) colnames(result$h) <- colnames(data)
    
    misc <- list(
      tol = result$tol,
      iter = result$iter,
      loss_type = loss,
      loss = result$loss,
      runtime = difftime(Sys.time(), start_time, units = "secs"),
      w_init = w_init_mat,
      rank_search = list(
        k_optimal = rank_result$k_optimal,
        k_low = rank_result$k_low,
        k_high = rank_result$k_high,
        overfitting_detected = rank_result$overfitting_detected
      )
    )
    
    return(
      new("nmf", w = result$w, d = result$d, h = result$h, misc = misc))
  }
  
  # --- Multi-rank CV: loop over (rank, replicate) and return data frame ---
  if (length(k_vec) > 1) {
    reps <- length(cv_seeds)
    n_results <- length(k_vec) * reps
    results <- vector("list", n_results)
    idx <- 1L
    
    for (rep in seq_len(reps)) {
      rep_cv_seed <- cv_seeds[rep]
      for (rank in k_vec) {
        # Use deterministic seed derived from cv_seed and rank.
        init_seed <- as.integer((rep_cv_seed + rank) %% .Machine$integer.max)
        
        result_i <- Rcpp_nmf_full(
          A_sexp = data, k_vec = as.integer(rank), k_auto = FALSE,
          k_auto_min = 2L, k_auto_max = 50L,
          L1 = as.double(L1), L2 = as.double(L2),
          L21 = as.double(L21), angular = as.double(angular),
          upper_bound = as.double(upper_bound),
          graph_lambda = as.double(graph_lambda),
          tol = tol, maxit = as.integer(maxit),
          cd_maxit = as.integer(cd_maxit), cd_tol = cd_tol,
          threads = as.integer(threads),
          seed = init_seed,
          nonneg_w = nonneg[1], nonneg_h = nonneg[2],
          sort_model = sort_model, mask_zeros = mask_zeros,
          track_loss_history = TRUE, track_train_loss = track_train_loss,
          loss_str = loss, huber_delta = huber_delta,
          holdout_fraction = test_fraction,
          cv_seeds = as.integer(rep_cv_seed),
          cv_patience = as.integer(patience),
          resource_override = resource,
          norm_str = norm,
          graph_W_sexp = graph_W, graph_H_sexp = graph_H,
          w_init_sexp = NULL,
          h_init_sexp = NULL,
          verbose = FALSE,
          mask_sexp = if (!is.null(mask) && inherits(mask, "sparseMatrix")) mask else NULL,
          projective = projective,
          symmetric_nmf = symmetric,
          solver_mode = solver_mode,
          init_mode = init_mode,
          irls_max_iter = as.integer(irls_max_iter),

          irls_tol = irls_tol,
          gp_dispersion_mode = dispersion_mode,
          gp_theta_init = theta_init,
          gp_theta_max = theta_max,
          zi_mode = zi_mode,
          zi_em_iters = as.integer(zi_em_iters),
          gp_theta_min = theta_min,
          nb_size_init = nb_size_init,
          nb_size_max = nb_size_max,
          nb_size_min = nb_size_min,
          gamma_phi_init = gamma_phi_init,
          gamma_phi_max = gamma_phi_max,
          gamma_phi_min = gamma_phi_min,
          robust_delta = robust_delta,
          tweedie_power = tweedie_power,
          target_H_sexp = target_H,
          target_lambda = target_lambda,
          enable_profiling = profile
        )
        
        results[[idx]] <- data.frame(
          rep = rep,
          k = rank,
          train_mse = if (!is.null(result_i$train_loss)) result_i$train_loss else NA_real_,
          test_mse = if (!is.null(result_i$best_test_loss)) result_i$best_test_loss else NA_real_,
          best_iter = if (!is.null(result_i$best_iter)) result_i$best_iter + 1L else NA_integer_,
          total_iter = if (!is.null(result_i$iter)) result_i$iter else NA_integer_,
          mean_theta = if (!is.null(result_i$theta)) mean(result_i$theta) else NA_real_,
          mean_dispersion = if (!is.null(result_i$dispersion)) mean(result_i$dispersion) else NA_real_
        )
        idx <- idx + 1L
      }
    }
    
    df <- do.call(rbind, results)
    class(df) <- c("nmfCrossValidate", "data.frame")
    df$rep <- as.factor(df$rep)
    return(df)
  }
  
  # --- Resource override ---
  resource_override <- resource
  
  # --- Zero-copy GPU NMF for gpu_sparse_matrix (data already on device) ---
  # This path cannot go through Rcpp_nmf_full() because the data is GPU-resident.
  if (is_gpu_sparse && length(k_vec) == 1 && test_fraction == 0) {
    if (verbose) message("[RcppML] Using GPU zero-copy backend for NMF")
    
    gpu_result <- .gpu_nmf_zerocopy(
      gpu_mat = data, k = k_vec[1], maxit = maxit, tol = tol, seed = seed_int,
      L1 = as.double(L1), L2 = as.double(L2),
      L21 = as.double(L21), ortho = as.double(angular),
      upper_bound = as.double(upper_bound),
      cd_maxit = as.integer(cd_maxit), verbose = verbose,
      nonneg = nonneg,
      w_init = w_init_mat,
      loss = loss, huber_delta = huber_delta,
      irls_max_iter = as.integer(irls_max_iter),
      irls_tol = as.double(irls_tol),
      norm = norm
    )
    
    k_final <- ncol(gpu_result$w)
    colnames(gpu_result$w) <- rownames(gpu_result$h) <- paste0("nmf", 1:k_final)
    
    misc <- list(
      tol = gpu_result$tol,
      iter = gpu_result$iter,
      loss_type = loss,
      loss = gpu_result$loss,
      runtime = difftime(Sys.time(), start_time, units = "secs"),
      w_init = w_init_mat,
      backend = "gpu"
    )
    
    return(
      new("nmf", w = gpu_result$w, d = gpu_result$d, h = gpu_result$h, misc = misc))
  }
  
  # --- Call C++ full gateway (handles CPU and GPU via resource_override) ---
  if (is_gpu_sparse) {
    stop("gpu_sparse_matrix only supports single-rank GPU NMF (no CV, no multi-rank). ",
         "Use sp_read() for CPU-side loading, or convert to dgCMatrix.",
         call. = FALSE)
  }
  result <- Rcpp_nmf_full(
    A_sexp = data,
    k_vec = k_vec,
    k_auto = k_auto,
    k_auto_min = as.integer(k_auto_min),
    k_auto_max = as.integer(k_auto_max),
    L1 = as.double(L1),
    L2 = as.double(L2),
    L21 = as.double(L21),
    angular = as.double(angular),
    upper_bound = as.double(upper_bound),
    graph_lambda = as.double(graph_lambda),
    tol = tol,
    maxit = as.integer(maxit),
    cd_maxit = as.integer(cd_maxit),
    cd_tol = cd_tol,
    threads = as.integer(threads),
    seed = as.integer(seed_int),
    nonneg_w = nonneg[1],
    nonneg_h = nonneg[2],
    sort_model = sort_model,
    mask_zeros = mask_zeros,
    track_loss_history = TRUE,
    track_train_loss = track_train_loss,
    loss_str = loss,
    huber_delta = huber_delta,
    holdout_fraction = test_fraction,
    cv_seeds = cv_seeds,
    cv_patience = as.integer(patience),
    resource_override = resource_override,
    norm_str = norm,
    graph_W_sexp = graph_W,
    graph_H_sexp = graph_H,
    w_init_sexp = w_init_mat,
    h_init_sexp = h_init,
    verbose = verbose,
    mask_sexp = if (!is.null(mask) && inherits(mask, "sparseMatrix")) mask else NULL,
    projective = projective,
    symmetric_nmf = symmetric,
    solver_mode = solver_mode,
    init_mode = init_mode,
    irls_max_iter = as.integer(irls_max_iter),

    irls_tol = irls_tol,
    gp_dispersion_mode = dispersion_mode,
    gp_theta_init = theta_init,
    gp_theta_max = theta_max,
    zi_mode = zi_mode,
    zi_em_iters = as.integer(zi_em_iters),
    gp_theta_min = theta_min,
    nb_size_init = nb_size_init,
    nb_size_max = nb_size_max,
    nb_size_min = nb_size_min,
    gamma_phi_init = gamma_phi_init,
    gamma_phi_max = gamma_phi_max,
    gamma_phi_min = gamma_phi_min,
    robust_delta = robust_delta,
    tweedie_power = tweedie_power,
    on_iteration_r = on_iteration,
    target_H_sexp = target_H,
    target_lambda = target_lambda,
    enable_profiling = profile
  )
  
  mode <- attr(result, "mode")
  
  # Surface resource diagnostics when verbose
  diag <- attr(result, "diagnostics")
  if (verbose && !is.null(diag)) {
    message(sprintf("[RcppML] Resource: %s (%s)", diag$plan, diag$reason))
  }
  
  # ---------------------------------------------------------------
  # Multi-rank CV → data.frame
  # ---------------------------------------------------------------
  if (identical(mode, "multi_cv")) {
    df <- data.frame(
      rep = result$rep,
      k = result$k,
      train_mse = result$train_mse,
      test_mse = result$test_mse,
      best_iter = result$best_iter,
      total_iter = result$total_iter
    )
    class(df) <- c("nmfCrossValidate", "data.frame")
    df$rep <- as.factor(df$rep)
    return(df)
  }
  
  # ---------------------------------------------------------------
  # Single-rank or auto-rank → S4 object
  # ---------------------------------------------------------------
  
  # Dimnames
  k_final <- ncol(result$w)
  colnames(result$w) <- rownames(result$h) <- paste0("nmf", 1:k_final)
  if (!is.null(rownames(data))) rownames(result$w) <- rownames(data)
  if (!is.null(colnames(data))) colnames(result$h) <- colnames(data)
  
  # Build misc
  misc <- list(
    tol = result$tol,
    iter = result$iter,
    loss_type = loss,
    runtime = difftime(Sys.time(), start_time, units = "secs"),
    w_init = if (!is.null(w_init_mat)) w_init_mat else NULL,
    solver_mode = solver_mode,
    L1 = L1,
    L2 = L2,
    nonneg = nonneg,
    upper_bound = upper_bound
  )
  
  # CV outputs
  if (test_fraction > 0 && !k_auto) {
    misc$train_loss <- result$train_loss
    misc$test_loss <- result$best_test_loss
    misc$best_iter <- result$best_iter
    misc$loss <- result$train_loss
  } else {
    misc$loss <- result$loss
  }
  
  # Loss history
  if (!is.null(result$train_loss_history)) misc$loss_history <- result$train_loss_history
  if (!is.null(result$test_loss_history)) misc$test_loss_history <- result$test_loss_history
  
  # Auto-rank metadata
  if (identical(mode, "auto_rank")) {
    misc$rank_search <- list(
      k_optimal = result$k_optimal,
      k_low = result$k_low,
      k_high = result$k_high,
      overfitting_detected = result$overfitting_detected
    )
  }
  
  # Multiple-init tracking
  if (!is.null(result$all_init_losses)) {
    misc$all_inits <- data.frame(
      init = seq_along(result$all_init_losses),
      loss = result$all_init_losses,
      selected = seq_along(result$all_init_losses) == result$best_init_idx
    )
  }
  
  # GP theta parameters
  if (!is.null(result$theta)) {
    misc$theta <- result$theta
    if (!is.null(rownames(data))) names(misc$theta) <- rownames(data)
  }
  
  # Gamma/InvGauss dispersion parameters
  if (!is.null(result$dispersion)) {
    misc$dispersion <- result$dispersion
    if (!is.null(rownames(data))) names(misc$dispersion) <- rownames(data)
  }
  
  # ZIGP pi parameters
  if (!is.null(result$pi_row)) {
    misc$pi_row <- result$pi_row
    if (!is.null(rownames(data))) names(misc$pi_row) <- rownames(data)
  }
  if (!is.null(result$pi_col)) {
    misc$pi_col <- result$pi_col
    if (!is.null(colnames(data))) names(misc$pi_col) <- colnames(data)
  }

  # Profiling data
  if (!is.null(result$profile)) {
    misc$profile <- result$profile
  }
  
  new("nmf", w = result$w, d = result$d, h = result$h, misc = misc)
}
