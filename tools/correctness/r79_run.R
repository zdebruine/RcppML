lib <- commandArgs(TRUE)[1]
library(Matrix); library(RcppML, lib.loc = lib)
ver <- packageVersion("RcppML", lib.loc = lib)
cat("RcppML", as.character(ver), "\n")
A <- readRDS(file.path(Sys.getenv("RW"), "out/pbmc3k_lognorm.rds"))
uniq <- function(w) { nz <- w > 0; colSums(nz[rowSums(nz) == 1, , drop = FALSE]) }
run <- function(label, ...) {
  t <- system.time(m <- tryCatch(RcppML::nmf(A, k = 21, seed = 12, tol = 1e-4, maxit = 1000, ...), error = function(e) e))[3]
  if (inherits(m, "error")) { cat(label, "ERROR", conditionMessage(m), "\n"); return(invisible()) }
  s4 <- isS4(m)
  w <- if (s4) m@w else m$w; h <- if (s4) m@h else m$h; d <- if (s4) m@d else m$d
  it <- tryCatch(if (s4) m@misc$iter else m$iter, error = function(e) NA)
  R <- w %*% (d * h); mse <- (sum(R^2) - 2 * sum(A * R) + sum(A@x^2)) / prod(dim(A))
  u <- uniq(w)
  cat(sprintf("%-26s iter=%s time=%.1fs mse=%.6f W_sparsity=%.3f rows_nonzero_in_one_factor: total=%d median_per_factor=%g range=%d-%d\n",
              label, paste(it, collapse = ","), t, mse, mean(w == 0), sum(u), median(u), min(u), max(u)))
}
run("defaults")
if (ver >= "1.0.0") {
  run("solver=cd", solver = "cd")
  run("solver=cholesky", solver = "cholesky")
  run("cd tight", solver = "cd", cd_tol = 1e-8, cd_maxit = 100)
}
