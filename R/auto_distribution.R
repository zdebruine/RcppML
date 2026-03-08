#' @title Auto-select NMF distribution
#'
#' @description Fits NMF with multiple loss functions (distributions) and selects
#' the best based on per-element AIC/BIC. Useful for determining whether count data
#' is best modeled with Gaussian (MSE), Poisson/GP, or Negative Binomial loss.
#'
#' @param data Input matrix (dense or sparse dgCMatrix)
#' @param k Factorization rank
#' @param distributions Character vector of distributions to compare.
#'   Default: \code{c("mse", "gp", "nb")}
#' @param criterion Selection criterion: \code{"bic"} (default) or \code{"aic"}
#' @param maxit Maximum iterations per fit
#' @param seed Random seed for reproducibility
#' @param verbose Print progress and comparison table
#' @param ... Additional arguments passed to \code{\link{nmf}}
#'
#' @return A list with:
#' \describe{
#'   \item{best}{Character string: name of the best distribution}
#'   \item{comparison}{Data frame with distribution, nll, df, aic, bic, selected}
#'   \item{models}{Named list of fitted nmf objects}
#' }
#'
#' @details
#' For each distribution, NMF is fit and the final negative log-likelihood (NLL)
#' is computed. For GP and NB, the C++ loss is already the total NLL.
#' For MSE (Gaussian), the C++ loss is the sum of squared errors (SSE), which is
#' converted to Gaussian NLL: \eqn{\text{NLL} = (N/2)(1 + \log(2\pi \cdot \text{SSE}/N))}.
#'
#' The number of effective parameters is:
#' \itemize{
#'   \item \code{mse}: \eqn{k(m + n) + 1} (factor params + noise variance)
#'   \item \code{gp}: \eqn{k(m + n) + m} (factor params + dispersion per row)
#'   \item \code{nb}: \eqn{k(m + n) + m} (factor params + size per row)
#' }
#'
#' BIC = \eqn{2 \times \text{NLL} + \text{df} \times \log(N)} where \eqn{N} is
#' the number of observations (nonzeros for sparse, all entries for dense).
#' AIC = \eqn{2 \times \text{NLL} + 2 \times \text{df}}.
#'
#' @examples
#' \dontrun{
#' library(Matrix)
#' set.seed(42)
#' A <- abs(rsparsematrix(100, 50, 0.3))
#' result <- auto_nmf_distribution(A, k = 5)
#' print(result$comparison)
#' cat("Best distribution:", result$best, "\n")
#' }
#'
#' @seealso \code{\link{score_test_distribution}}, \code{\link{diagnose_zero_inflation}},
#'   \code{\link{nmf}}
#' @export
auto_nmf_distribution <- function(data, k,
                                   distributions = c("mse", "gp", "nb"),
                                   criterion = c("bic", "aic"),
                                   maxit = 50,
                                   seed = NULL,
                                   verbose = FALSE,
                                   ...) {
  criterion <- match.arg(criterion)
  distributions <- match.arg(distributions, c("mse", "gp", "nb", "mae", "huber"),
                              several.ok = TRUE)

  m <- nrow(data)
  n <- ncol(data)
  is_sparse <- inherits(data, "dgCMatrix") || inherits(data, "sparseMatrix")

  # Number of observations
  N <- if (is_sparse) Matrix::nnzero(data) else as.numeric(m) * n

  models <- list()
  results <- data.frame(
    distribution = character(),
    nll = numeric(),
    df = integer(),
    aic = numeric(),
    bic = numeric(),
    stringsAsFactors = FALSE
  )

  for (dist in distributions) {
    if (verbose) cat("Fitting NMF with loss =", dist, "...\n")

    # Fit model
    model <- nmf(data, k, loss = dist, maxit = maxit, seed = seed,
                 verbose = FALSE, ...)
    models[[dist]] <- model

    # Number of effective parameters
    n_factor_params <- k * (m + n)  # W (m×k) + H (k×n)
    df <- if (dist == "mse") {
      n_factor_params + 1   # + noise variance sigma^2
    } else if (dist %in% c("gp", "nb")) {
      n_factor_params + m   # + dispersion per row
    } else {
      n_factor_params       # MAE/Huber: no extra params (approximate)
    }

    # NLL computation
    raw_loss <- model@misc$loss
    if (dist == "mse") {
      # MSE loss is SSE = sum((y - mu)^2). Convert to Gaussian NLL:
      # NLL = (N/2)(1 + log(2*pi*SSE/N))
      nll <- (N / 2) * (1 + log(2 * pi * raw_loss / N))
    } else {
      # GP and NB losses are already proper NLL (sum of per-element NLL)
      nll <- raw_loss
    }

    aic <- 2 * nll + 2 * df
    bic <- 2 * nll + df * log(N)

    results <- rbind(results, data.frame(
      distribution = dist,
      nll = nll,
      df = df,
      aic = aic,
      bic = bic,
      stringsAsFactors = FALSE
    ))
  }

  # Select best
  score_col <- if (criterion == "bic") "bic" else "aic"
  best_idx <- which.min(results[[score_col]])
  best_dist <- results$distribution[best_idx]
  results$selected <- results$distribution == best_dist

  if (verbose) {
    cat("\n--- Distribution Comparison (", toupper(criterion), ") ---\n")
    print(results, row.names = FALSE)
    cat("\nBest distribution:", best_dist, "\n")
  }

  list(
    best = best_dist,
    comparison = results,
    models = models
  )
}


#' @title Score-test distribution diagnostic
#'
#' @description Given a baseline MSE-fitted NMF model and the original data,
#' computes score-test statistics for the power-variance family
#' (\eqn{V(\mu) = \mu^p}) to determine the best-fitting distribution without
#' refitting. Optionally tests for NB overdispersion.
#'
#' @param data Original data matrix (sparse or dense)
#' @param model A fitted NMF object (from \code{nmf()} with \code{loss="mse"})
#' @param powers Numeric vector of variance powers to test.
#'   Default \code{c(0, 1, 2, 3)} covers Gaussian, Poisson, Gamma, Inverse Gaussian.
#' @param test_nb Logical; if \code{TRUE} and data is integer-valued, also test
#'   the NB overdispersion diagnostic.
#' @param min_mu Floor for predicted values to avoid division by zero. Default 1e-6.
#'
#' @return A list with:
#' \describe{
#'   \item{scores}{Data frame with columns \code{power}, \code{T_stat}, \code{abs_T},
#'     \code{distribution} (label)}
#'   \item{best_power}{Numeric: the power p with smallest \code{|T_p|}}
#'   \item{best_distribution}{Character: name of the best-matching distribution}
#'   \item{nb_diagnostic}{If \code{test_nb=TRUE}: list with \code{T_NB} and
#'     \code{overdispersed} (logical)}
#' }
#'
#' @details
#' The score test statistic for variance power \eqn{p} is:
#' \deqn{T_p = \text{mean}\left(\frac{r_{ij}^2}{\mu_{ij}^p} - 1\right)}
#' where \eqn{r_{ij} = x_{ij} - \mu_{ij}} are residuals and
#' \eqn{\mu_{ij} = (WH)_{ij}} are predicted values.
#'
#' Under the correct model, \eqn{E[T_p] = 0}. The power minimizing \eqn{|T_p|}
#' best matches the observed variance-mean relationship.
#'
#' The NB diagnostic tests for quadratic overdispersion:
#' \deqn{T_{NB} = \text{mean}\left(\frac{r_{ij}^2 - \mu_{ij}}{\mu_{ij}^2}\right)}
#' If \eqn{T_{NB} > 0.1}, there is substantial overdispersion beyond Poisson,
#' suggesting NB may be preferable to GP.
#'
#' @examples
#' \dontrun{
#' A <- abs(rsparsematrix(200, 100, 0.3))
#' model <- nmf(A, k = 5, loss = "mse")
#' diag <- score_test_distribution(A, model)
#' print(diag$scores)
#' cat("Best distribution:", diag$best_distribution, "\n")
#' }
#'
#' @seealso \code{\link{auto_nmf_distribution}}, \code{\link{nmf}}
#' @export
score_test_distribution <- function(data, model,
                                     powers = c(0, 1, 2, 3),
                                     test_nb = TRUE,
                                     min_mu = 1e-6) {
  # Reconstruct W * diag(d) * H
  W <- model@w
  H <- model@h
  d <- model@d
  WdH <- (W %*% diag(d)) %*% H

  is_sparse <- inherits(data, "dgCMatrix") || inherits(data, "sparseMatrix")

  # Power label map
  power_labels <- c("0" = "gaussian", "1" = "gp", "2" = "gamma", "3" = "inverse_gaussian")

  if (is_sparse) {
    # Only iterate over nonzero entries
    data_csc <- as(data, "dgCMatrix")
    idx <- Matrix::which(data_csc != 0, arr.ind = TRUE)
    x_obs <- data_csc[idx]
    mu_obs <- pmax(WdH[idx], min_mu)
    r_obs <- x_obs - mu_obs
  } else {
    x_obs <- as.vector(data)
    mu_obs <- pmax(as.vector(WdH), min_mu)
    r_obs <- x_obs - mu_obs
  }

  r2 <- r_obs^2
  scores <- data.frame(
    power = powers,
    T_stat = numeric(length(powers)),
    abs_T = numeric(length(powers)),
    distribution = character(length(powers)),
    stringsAsFactors = FALSE
  )

  for (i in seq_along(powers)) {
    p <- powers[i]
    T_p <- mean(r2 / (mu_obs^p) - 1)
    scores$T_stat[i] <- T_p
    scores$abs_T[i] <- abs(T_p)
    scores$distribution[i] <- if (as.character(p) %in% names(power_labels)) {
      power_labels[as.character(p)]
    } else {
      paste0("power_", p)
    }
  }

  best_idx <- which.min(scores$abs_T)
  best_power <- scores$power[best_idx]
  best_dist <- scores$distribution[best_idx]

  result <- list(
    scores = scores,
    best_power = best_power,
    best_distribution = best_dist
  )

  # NB overdispersion diagnostic for integer data
  if (test_nb) {
    is_integer <- all(x_obs == round(x_obs))
    if (is_integer) {
      T_NB <- mean((r2 - mu_obs) / mu_obs^2)
      result$nb_diagnostic <- list(
        T_NB = T_NB,
        overdispersed = T_NB > 0.1
      )
    }
  }

  result
}


#' @title Diagnose zero inflation
#'
#' @description Tests whether a dataset has excess zeros beyond what the chosen
#' distribution predicts, and recommends a zero-inflation mode.
#'
#' @param data Input matrix (sparse or dense)
#' @param model A fitted NMF model (any distribution)
#' @param threshold Minimum excess zero fraction to declare ZI. Default 0.05.
#'
#' @return A list with:
#' \describe{
#'   \item{excess_zero_rate}{Fraction of zeros exceeding the expected count}
#'   \item{has_zi}{Logical: TRUE if excess_zero_rate > threshold}
#'   \item{zi_mode}{Recommended mode: "none", "row", "col", or "twoway"}
#'   \item{row_excess}{Per-row excess zero rates}
#'   \item{col_excess}{Per-col excess zero rates}
#' }
#'
#' @details
#' Computes the expected number of zeros under the fitted distribution
#' (Poisson approximation: \eqn{P(X=0) \approx e^{-\mu}}), compares to
#' the observed zero count, and recommends ZI if the excess is large.
#'
#' ZI granularity is determined by whether per-row and per-col excess rates
#' have high variance (suggesting different rows/cols have different ZI levels).
#'
#' @seealso \code{\link{auto_nmf_distribution}}, \code{\link{nmf}}
#' @export
diagnose_zero_inflation <- function(data, model, threshold = 0.05) {
  W <- model@w
  H <- model@h
  d <- model@d
  mu_mat <- (W %*% diag(d)) %*% H

  m <- nrow(data)
  n <- ncol(data)
  is_sparse <- inherits(data, "dgCMatrix") || inherits(data, "sparseMatrix")

  # Count observed zeros per row and column
  if (is_sparse) {
    data_csc <- as(data, "dgCMatrix")
    obs_nz_per_col <- diff(data_csc@p)
    obs_zeros_per_col <- m - obs_nz_per_col
    # Per-row: need CSR or count manually
    row_nz <- tabulate(data_csc@i + 1L, nbins = m)
    obs_zeros_per_row <- n - row_nz
  } else {
    obs_zeros_per_row <- rowSums(data == 0)
    obs_zeros_per_col <- colSums(data == 0)
  }

  # Expected zeros under Poisson model: P(X=0) = exp(-mu)
  # (This is an approximation; for GP/NB, the exact rate differs, but
  # Poisson serves as a baseline diagnostic.)
  mu_floor <- pmax(as.matrix(mu_mat), 1e-8)
  expected_zero_prob <- exp(-mu_floor)
  expected_zeros_per_row <- rowSums(expected_zero_prob)
  expected_zeros_per_col <- colSums(expected_zero_prob)

  # Excess zero rates
  row_excess <- pmax(0, (obs_zeros_per_row - expected_zeros_per_row) / n)
  col_excess <- pmax(0, (obs_zeros_per_col - expected_zeros_per_col) / m)
  global_excess <- mean(c(row_excess, col_excess))

  has_zi <- global_excess > threshold

  # Determine granularity
  if (!has_zi) {
    zi_mode <- "none"
  } else {
    row_var <- var(row_excess)
    col_var <- var(col_excess)
    row_structured <- row_var > 0.001
    col_structured <- col_var > 0.001
    if (row_structured && col_structured) {
      zi_mode <- "twoway"
    } else if (col_structured) {
      zi_mode <- "col"
    } else {
      zi_mode <- "row"   # default: row-level ZI
    }
  }

  list(
    excess_zero_rate = global_excess,
    has_zi = has_zi,
    zi_mode = zi_mode,
    row_excess = row_excess,
    col_excess = col_excess
  )
}


#' @title Diagnose dispersion mode
#'
#' @description Determines whether dispersion should be estimated per-row,
#' per-column, or globally by examining the coefficient of variation of
#' per-row and per-column dispersion estimates.
#'
#' @param data Input matrix (sparse or dense)
#' @param model A fitted NMF model
#' @param cv_threshold CV threshold for declaring structured dispersion.
#'   Default 0.5.
#' @param min_mu Floor for predicted values. Default 1e-6.
#'
#' @return A list with:
#' \describe{
#'   \item{mode}{Recommended DispersionMode: "global", "per_row", or "per_col"}
#'   \item{global_phi}{Global dispersion estimate}
#'   \item{row_cv}{CV of per-row dispersion estimates}
#'   \item{col_cv}{CV of per-col dispersion estimates}
#' }
#'
#' @details
#' Computes moment-based dispersion estimates \eqn{\hat{\phi} = r^2 / \mu^p}
#' (where \eqn{p} is determined by the distribution) per row and per column.
#' If the coefficient of variation (CV) of per-row estimates exceeds
#' \code{cv_threshold}, per-row dispersion is recommended; similarly for
#' per-column. If both CVs are low, global dispersion suffices.
#'
#' @seealso \code{\link{auto_nmf_distribution}}, \code{\link{nmf}}
#' @export
diagnose_dispersion <- function(data, model, cv_threshold = 0.5,
                                 min_mu = 1e-6) {
  W <- model@w
  H <- model@h
  d <- model@d
  mu_mat <- pmax(as.matrix((W %*% diag(d)) %*% H), min_mu)

  loss_type <- if (!is.null(model@misc$loss_type)) model@misc$loss_type else "mse"

  # Determine variance power from distribution
  p <- switch(loss_type,
    mse = 0, gaussian = 0,
    gp = 1, kl = 1,
    gamma = 2,
    inverse_gaussian = 3,
    nb = 1,  # NB has V(mu)=mu+mu^2/r, but mu^1 is a reasonable approx
    0  # default
  )

  r2 <- (as.matrix(data) - mu_mat)^2
  phi_elem <- r2 / (mu_mat^p)

  # Per-row and per-col dispersion estimates (trimmed mean to reduce outlier influence)
  row_phi <- apply(phi_elem, 1, function(x) mean(x, trim = 0.1))
  col_phi <- apply(phi_elem, 2, function(x) mean(x, trim = 0.1))
  global_phi <- mean(phi_elem, trim = 0.1)

  row_cv <- sd(row_phi) / mean(row_phi)
  col_cv <- sd(col_phi) / mean(col_phi)

  if (row_cv > cv_threshold && col_cv > cv_threshold) {
    # Both are structured — pick the one with higher CV
    mode <- if (row_cv >= col_cv) "per_row" else "per_col"
  } else if (row_cv > cv_threshold) {
    mode <- "per_row"
  } else if (col_cv > cv_threshold) {
    mode <- "per_col"
  } else {
    mode <- "global"
  }

  list(
    mode = mode,
    global_phi = global_phi,
    row_cv = row_cv,
    col_cv = col_cv
  )
}
