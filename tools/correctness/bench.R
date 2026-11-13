# Timing benchmark across RcppML versions.  Usage: Rscript bench.R <lib> <label>
# Fixed iteration counts, so times compare work per iteration; the loss column shows quality.
args <- commandArgs(TRUE); lib <- args[1]; label <- args[2]
suppressPackageStartupMessages({ library(Matrix); library(RcppML, lib.loc = lib) })
ver <- packageVersion("RcppML", lib.loc = lib); new_api <- ver >= "1.0.0"
RW <- Sys.getenv("RW"); thr <- as.integer(Sys.getenv("OMP_NUM_THREADS", "16"))
if (!new_api) options(RcppML.threads = thr, RcppML.verbose = FALSE)
rows <- list()
rec <- function(task, secs, quality = NA, note = "") {
  rows[[length(rows) + 1]] <<- data.frame(label = label, version = as.character(ver), task = task, seconds = round(secs, 3), quality = signif(quality, 6), note = note)
  cat(sprintf("%-10s %-44s %8.3fs  quality=%-12s %s\n", label, task, secs, format(signif(quality, 6)), note))
}
mse_s <- function(m, A) { s4 <- isS4(m); w <- if (s4) m@w else m$w; h <- if (s4) m@h else m$h; d <- if (s4) m@d else m$d
  R <- w %*% (d * h); (sum(R^2) - 2 * sum(A * R) + sum(A^2)) / prod(dim(A)) }
timeit <- function(task, expr, qual = function(m) NA, reps = 3) {
  best <- Inf; m <- NULL; err <- NULL
  for (i in seq_len(reps)) { t <- system.time(m <- tryCatch(expr(), error = function(e) { err <<- conditionMessage(e); NULL }))[3]; if (is.null(m)) break; best <- min(best, t) }
  if (is.null(m)) rec(task, NA, NA, paste("ERROR:", substr(err, 1, 80))) else rec(task, best, tryCatch(qual(m), error = function(e) NA))
}
pb <- readRDS(file.path(RW, "out/pbmc3k_lognorm.rds"))                       # 13713 x 2638 sparse, 6% dense
set.seed(1); big <- rsparsematrix(30000, 8000, 0.03, rand.x = function(n) rexp(n))   # 7.2M nonzeros
set.seed(2); dn <- matrix(rexp(3000 * 1500), 3000) * (matrix(runif(3000 * 1500), 3000) < 0.5)
cnt <- pb; cnt@x <- round(expm1(cnt@x) / 10 + 0.5)                           # count-like
data <- list(pbmc3k = pb, sparse30k = big, dense3k = dn)
for (dn_ in names(data)) { A <- data[[dn_]]
  for (k in c(8, 20, 48)) {
    base <- function(...) if (new_api) RcppML::nmf(A, k, seed = 1, tol = 0, maxit = 20, threads = thr, ...) else RcppML::nmf(A, k, seed = 1, tol = 1e-12, maxit = 20)
    q <- function(m) mse_s(m, A)
    if (!new_api) timeit(sprintf("nmf %s k=%d default", dn_, k), base, q) else for (sv in c("auto", "cd", "cholesky")) timeit(sprintf("nmf %s k=%d solver=%s", dn_, k, sv), function() base(solver = sv), q)
  }
}
# converged fits: wall time and quality at the default tolerance
for (k in c(10, 30)) timeit(sprintf("nmf pbmc3k k=%d converged (tol=1e-4)", k), function() if (new_api) RcppML::nmf(pb, k, seed = 1, tol = 1e-4, maxit = 200, threads = thr) else RcppML::nmf(pb, k, seed = 1, tol = 1e-4, maxit = 200), function(m) mse_s(m, pb), reps = 2)
if (new_api) {
  timeit("nmf pbmc3k k=20 L1=0.1", function() RcppML::nmf(pb, 20, seed = 1, tol = 0, maxit = 20, L1 = c(0.1, 0.1), threads = thr), function(m) mse_s(m, pb))
  for (ls in c("gp", "nb")) timeit(sprintf("nmf counts k=20 loss=%s sparse", ls), function() RcppML::nmf(cnt, 20, seed = 1, tol = 0, maxit = 10, loss = ls, threads = thr), reps = 2)
  timeit("nmf counts[1:3000,1:1000] k=10 loss=gp dense", function() RcppML::nmf(as.matrix(cnt[1:3000, 1:1000]), 10, seed = 1, tol = 0, maxit = 10, loss = "gp", threads = thr), reps = 2)
  timeit("cv pbmc3k k=20 test_fraction=0.1", function() RcppML::nmf(pb, 20, seed = 1, tol = 0, maxit = 20, test_fraction = 0.1, patience = 1000, threads = thr), reps = 2)
  timeit("cv pbmc3k k=4:12", function() RcppML::nmf(pb, 4:12, seed = 1, test_fraction = 0.1, maxit = 30, threads = thr), reps = 1)
  timeit("nmf pbmc3k k=20 custom mask 5%", { set.seed(3); M <- rsparsematrix(nrow(pb), ncol(pb), 0.05, rand.x = function(n) rep(1, n)); function() RcppML::nmf(pb, 20, seed = 1, tol = 0, maxit = 10, mask = M, threads = thr) }, function(m) mse_s(m, pb), reps = 2)
  for (meth in c("lanczos", "irlba", "randomized", "deflation", "krylov")) timeit(sprintf("svd pbmc3k k=20 %s", meth), function() RcppML::svd(pb, k = 20, method = meth, seed = 1, threads = thr), function(s) sum(s@d^2), reps = 2)
  timeit("pca pbmc3k k=20", function() RcppML::pca(pb, k = 20, seed = 1, threads = thr), function(s) sum(s@d^2), reps = 2)
  timeit("svd pbmc3k k=10 nonneg", function() RcppML::svd(pb, k = 10, nonneg = TRUE, seed = 1, threads = thr), function(s) sum(s@d^2), reps = 2)
}
timeit("dclust pbmc3k min_samples=50", function() if (new_api) RcppML::dclust(pb, min_samples = 50, seed = 1, threads = thr) else RcppML::dclust(pb, min_samples = 50, seed = 1), function(m) length(m), reps = 2)
timeit("bipartition pbmc3k", function() RcppML::bipartition(pb, seed = 1), function(m) m$dist, reps = 3)
set.seed(4); Wn <- matrix(runif(nrow(pb) * 20), ncol = 20)
if (new_api) timeit("nnls project pbmc3k k=20", function() RcppML::nnls(w = Wn, A = pb, threads = thr), function(h) sum(h)) else timeit("project pbmc3k k=20", function() RcppML::project(pb, w = Wn), function(h) sum(h))
write.csv(do.call(rbind, rows), file.path(RW, "out", sprintf("bench_%s.csv", label)), row.names = FALSE)
