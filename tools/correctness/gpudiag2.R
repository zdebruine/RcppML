# CPU and GPU against exact double-precision NNLS-ALS from the same W init, on a nearly exact low-rank matrix.
suppressPackageStartupMessages({ library(Matrix); library(RcppML) })
nnls_ref <- function(G, b, it = 5000, tol = 1e-14) { k <- length(b); x <- rep(0, k)
  for (s in seq_len(it)) { mx <- 0; for (i in seq_len(k)) { xi <- max(0, x[i] + (b[i] - sum(G[i, ] * x)) / G[i, i]); mx <- max(mx, abs(xi - x[i])); x[i] <- xi }; if (mx < tol * (1 + max(abs(x)))) break }; x }
als_ref <- function(A, W, iters) { out <- numeric(iters)
  for (t in seq_len(iters)) { G <- crossprod(W); B <- crossprod(W, A); H <- apply(B, 2, function(b) nnls_ref(G, b))
    G <- tcrossprod(H); B <- H %*% t(A); W <- t(apply(B, 2, function(b) nnls_ref(G, b))); out[t] <- mean((A - W %*% H)^2) }; out }
set.seed(77); m <- 300; n <- 1000; k <- 4
W <- matrix(abs(rnorm(m * k)), m, k); H <- matrix(abs(rnorm(k * n)), k, n); S <- W %*% H
noise <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.05)); noise@x <- abs(noise@x) * 0.01
A <- as.matrix(S + noise); set.seed(5); W0 <- matrix(runif(m * k), m, k)
its <- c(1, 2, 3, 5, 10, 20); ref <- als_ref(A, W0, 20)
mse_of <- function(r) mean((A - r@w %*% (r@d * r@h))^2)
for (inp in c("dense", "sparse")) { X <- if (inp == "dense") A else as(A, "CsparseMatrix")
  for (res in c("cpu", "gpu")) for (sv in c("cd", "cholesky")) {
    v <- sapply(its, function(it) mse_of(suppressWarnings(nmf(X, k, seed = W0, maxit = it, tol = 0, resource = res, solver = sv))))
    cat(sprintf("%-6s %-3s %-8s  %s\n", inp, res, sv, paste(sprintf("it%d=%.3e(%.3f)", its, v, v / ref[its]), collapse = " "))) } }
cat("exact double ALS:      ", paste(sprintf("it%d=%.3e", its, ref[its]), collapse = " "), "\n")
