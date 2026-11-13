# GPU versus CPU, iteration by iteration, from the same init. A healthy GPU path tracks CPU at every count.
suppressPackageStartupMessages({ library(Matrix); library(RcppML) })
mse_of <- function(m, A) mean((as.matrix(A) - m@w %*% (m@d * m@h))^2)
run <- function(label, A, k, ...) {
  for (it in c(1, 2, 5, 10, 20, 40)) {
    c0 <- suppressWarnings(nmf(A, k, seed = 10, maxit = it, tol = 0, resource = "cpu", ...))
    g0 <- suppressWarnings(nmf(A, k, seed = 10, maxit = it, tol = 0, resource = "gpu", ...))
    cat(sprintf("%-34s it=%2d cpu=%.6g gpu=%.6g ratio=%.4f\n", label, it, mse_of(c0, A), mse_of(g0, A), mse_of(g0, A) / mse_of(c0, A)))
  }
}
set.seed(77); m <- 300; n <- 1000; k <- 4
W <- matrix(abs(rnorm(m * k)), m, k); H <- matrix(abs(rnorm(k * n)), k, n); S <- W %*% H
noise <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.05)); noise@x <- abs(noise@x) * 0.01
A <- as.matrix(S + noise)
run("near-exact rank 4, dense input", A, 4)
B <- A * (matrix(runif(m * n), m) < 0.3)
run("same, 30% kept, sparse input", as(B, "CsparseMatrix"), 4)
run("same, 30% kept, sparse, k=16", as(B, "CsparseMatrix"), 16)
C <- pmax(S + matrix(rnorm(m * n, sd = 0.5), m), 0) * (matrix(runif(m * n), m) < 0.3)
run("noisy, 30% kept, sparse input", as(C, "CsparseMatrix"), 4)
run("noisy, 30% kept, sparse, L1=0.1", as(C, "CsparseMatrix"), 4, L1 = c(0.1, 0.1))
run("noisy, 30% kept, sparse, L2=0.1", as(C, "CsparseMatrix"), 4, L2 = c(0.1, 0.1))
big <- rsparsematrix(20000, 5000, 0.02, rand.x = function(n) rexp(n))
for (kk in c(8, 32)) { tc <- system.time(c0 <- nmf(big, kk, seed = 1, maxit = 20, tol = 0, resource = "cpu"))[3]; tg <- system.time(g0 <- suppressWarnings(nmf(big, kk, seed = 1, maxit = 20, tol = 0, resource = "gpu")))[3]
  cat(sprintf("timing 20000x5000 2%% k=%d: cpu %.2fs loss %.6g | gpu %.2fs loss %.6g\n", kk, tc, c0@misc$loss, tg, g0@misc$loss)) }
