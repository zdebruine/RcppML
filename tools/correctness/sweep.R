# Independent correctness sweep for RcppML. Every check compares the package against
# reference math written here in plain double-precision R. Usage:
#   Rscript sweep.R <section> <resource>     section in A B C D E F ; resource cpu|gpu
args <- commandArgs(TRUE)
SECTION <- args[1]; RES <- if (length(args) > 1) args[2] else "cpu"
suppressPackageStartupMessages({ library(Matrix); library(RcppML) })
options(RcppML.gpu = (RES == "gpu"), width = 200)
TAG <- Sys.getenv("SWEEP_TAG", ""); OUT <- file.path(Sys.getenv("RW"), "out", sprintf("sweep%s_%s_%s.csv", if (nzchar(TAG)) paste0("_", TAG) else "", SECTION, RES))
rows <- list()
options(error = function() { try(write.csv(do.call(rbind, rows), OUT, row.names = FALSE)); cat("CRASHED
"); traceback(2); q(status = 1) })
rec <- function(check, path, param, value, status, note = "") {
  rows[[length(rows) + 1]] <<- data.frame(section = SECTION, resource = RES, check = check, path = path,
    param = as.character(param), value = if (is.numeric(value)) signif(value, 5) else NA_real_,
    status = status, note = substr(as.character(note), 1, 240), stringsAsFactors = FALSE)
  cat(sprintf("%-4s| %-26s| %-22s| %-22s| %-12s| %s %s\n", status, check, path, param,
              if (is.numeric(value)) format(signif(value, 5)) else "", note, ""))
}
try_ <- function(expr) tryCatch(expr, error = function(e) structure(conditionMessage(e), class = "sweep_err"))
is_err <- function(x) inherits(x, "sweep_err")
fit <- function(A, k, ...) try_(suppressWarnings(RcppML::nmf(A, k, resource = RES, ...)))
recon <- function(m) m@w %*% (m@d * m@h)
mse_of <- function(m, A) mean((as.matrix(A) - recon(m))^2)

# ---- reference solvers (double precision) -----------------------------------------------
nnls_ref <- function(G, b, lower = 0, upper = Inf, l1 = 0, x = NULL, it = 20000, tol = 1e-13) {
  k <- length(b); if (is.null(x)) x <- rep(0, k); b <- b - l1
  for (s in seq_len(it)) { mx <- 0
    for (i in seq_len(k)) { if (G[i, i] <= 0) next
      xi <- x[i] + (b[i] - sum(G[i, ] * x)) / G[i, i]; xi <- min(max(xi, lower), upper)
      mx <- max(mx, abs(xi - x[i])); x[i] <- xi }
    if (mx < tol * (1 + max(abs(x)))) break }
  x
}
obj_ls <- function(G, b, x) 0.5 * sum(x * (G %*% x)) - sum(b * x)
# optimality gap of factor H given W for min ||A - W H||: mean over sampled columns of
# (f(h) - f(h*)) / (||a||^2/2), plus worst KKT violation
gap_H <- function(W, H, A, ncols = 40) {
  A <- as.matrix(A); G <- crossprod(W); cols <- unique(round(seq(1, ncol(A), length.out = ncols)))
  g <- sapply(cols, function(j) { b <- drop(crossprod(W, A[, j])); hs <- nnls_ref(G, b)
    (obj_ls(G, b, H[, j]) - obj_ls(G, b, hs)) / (0.5 * sum(A[, j]^2) + 1e-300) })
  c(mean = mean(g), max = max(g))
}
als_ref <- function(A, W0, iters, order = c("H", "W")) {  # exact NNLS-ALS in double precision
  A <- as.matrix(A); W <- W0; H <- NULL; out <- numeric(iters)
  for (t in seq_len(iters)) {
    G <- crossprod(W); B <- crossprod(W, A); H <- sapply(seq_len(ncol(A)), function(j) nnls_ref(G, B[, j], it = 500, tol = 1e-10))
    G <- tcrossprod(H); B <- H %*% t(A); W <- t(sapply(seq_len(nrow(A)), function(i) nnls_ref(G, B[, i], it = 500, tol = 1e-10)))
    out[t] <- mean((A - W %*% H)^2) }
  out
}
sim <- function(m, n, k, noise = 0.1, seed = 1, sparse_frac = 0) {
  set.seed(seed); W <- matrix(rexp(m * k) * (runif(m * k) < 0.4), m, k); H <- matrix(rexp(k * n) * (runif(k * n) < 0.5), k, n)
  A <- W %*% H; A <- pmax(A + noise * mean(A) * matrix(rnorm(m * n), m, n), 0)
  if (sparse_frac > 0) A[matrix(runif(m * n) < sparse_frac, m, n)] <- 0
  list(A = A, W = W, H = H)
}
lapl <- function(G) as(Diagonal(x = rowSums(G)) - G, "CsparseMatrix")
chain <- function(n) { G <- bandSparse(n, k = c(-1, 1)); as(G * 1, "CsparseMatrix") }
cosmean <- function(M) { M <- as.matrix(M); Mn <- sweep(M, 2, sqrt(colSums(M^2)) + 1e-30, "/"); C <- crossprod(Mn); mean(abs(C[upper.tri(C)])) }  # M columns = factors
lap_energy <- function(F, G) { L <- Diagonal(x = rowSums(G)) - G; Fn <- F / sqrt(rowSums(F^2) + 1e-30); sum(Fn * (Fn %*% L)) }  # F rows = factors

# =========================================================================================
if (SECTION == "A") {  # NMF solver exactness, loss, determinism
  for (shape in list(c(300, 200, 5), c(600, 400, 12), c(400, 300, 40))) {
    m <- shape[1]; n <- shape[2]; k <- shape[3]; S <- sim(m, n, k, noise = 0.2, seed = k)
    for (inp in c("dense", "sparse")) { A <- if (inp == "dense") S$A else as(S$A, "CsparseMatrix")
      for (sv in c("auto", "cd", "cholesky")) { path <- sprintf("%s k=%d %s", inp, k, sv)
        r <- fit(A, k, seed = 7, tol = 1e-6, maxit = 60, solver = sv)
        if (is_err(r)) { rec("fit", path, "", NA, "ERR", r); next }
        e <- mse_of(r, S$A)
        rec("reported_loss_vs_explicit", path, "rel diff", abs(r@misc$loss - e) / e, if (min(abs(r@misc$loss - e) / e, abs(r@misc$loss / length(S$A) - e) / e) < 5e-3) "PASS" else "FAIL", sprintf("reported=%.6g explicit=%.6g", r@misc$loss, e))
        gh <- gap_H(r@w %*% diag(r@d), r@h, S$A); gw <- gap_H(t(r@d * r@h), t(r@w), t(S$A))
        best <- min(gh["mean"], gw["mean"])
        rec("halfstep_optimality_gap", path, "min(H|W, W|H)", best, if (best < 1e-5) "PASS" else "FAIL", sprintf("H|W mean=%.2e max=%.2e; W|H mean=%.2e max=%.2e", gh[1], gh[2], gw[1], gw[2]))
        rec("final_mse", path, "mse", e, "INFO", sprintf("iter=%s", r@misc$iter))
      }
    }
  }
  # iteration-by-iteration agreement with exact ALS from the same W init; monotonicity
  S <- sim(120, 80, 6, noise = 0.2, seed = 3); set.seed(11); W0 <- matrix(runif(120 * 6), 120, 6)
  ref <- als_ref(S$A, W0, 12)
  for (sv in c("cd", "cholesky", "auto")) for (inp in c("dense", "sparse")) {
    A <- if (inp == "dense") S$A else as(S$A, "CsparseMatrix")
    tr <- sapply(1:12, function(it) { r <- fit(A, 6, seed = W0, tol = 0, maxit = it, solver = sv); if (is_err(r)) NA else mse_of(r, S$A) })
    path <- sprintf("%s %s", inp, sv)
    if (anyNA(tr)) { rec("trajectory_vs_exact_ALS", path, "", NA, "ERR", "fit failed"); next }
    rec("trajectory_vs_exact_ALS", path, "max rel dev over 12 it", max(abs(tr - ref) / ref), if (max(abs(tr - ref) / ref) < 0.02) "PASS" else "FAIL",
        sprintf("it12: pkg=%.5g exactALS=%.5g; it1: pkg=%.5g exact=%.5g", tr[12], ref[12], tr[1], ref[1]))
    rec("loss_monotone", path, "max increase/loss", max(diff(tr)) / tr[12], if (max(diff(tr)) <= 1e-6 * tr[12]) "PASS" else "FAIL", paste(signif(tr, 4), collapse = " "))
  }
  # determinism: same seed, thread counts, sparse vs dense
  S <- sim(400, 300, 8, noise = 0.2, seed = 5); As <- as(S$A, "CsparseMatrix")
  for (sv in c("cd", "cholesky")) {
    a <- fit(As, 8, seed = 1, solver = sv, threads = 1); b <- fit(As, 8, seed = 1, solver = sv, threads = 1); c8 <- fit(As, 8, seed = 1, solver = sv, threads = 8); dn <- fit(S$A, 8, seed = 1, solver = sv, threads = 1)
    if (is_err(a) || is_err(c8) || is_err(dn)) { rec("determinism", sv, "", NA, "ERR", paste(a, c8, dn)[1]); next }
    rec("same_seed_repeat", sv, "max|dW|", max(abs(a@w - b@w)), if (max(abs(a@w - b@w)) == 0) "PASS" else "FAIL")
    rec("threads_1_vs_8", sv, "max|dW|", max(abs(a@w - c8@w)), if (max(abs(a@w - c8@w)) < 1e-4) "PASS" else "FAIL", sprintf("iter %s vs %s", a@misc$iter, c8@misc$iter))
    rec("sparse_vs_dense_input", sv, "rel mse diff", abs(mse_of(a, S$A) - mse_of(dn, S$A)) / mse_of(a, S$A), if (abs(mse_of(a, S$A) - mse_of(dn, S$A)) / mse_of(a, S$A) < 1e-3) "PASS" else "FAIL", sprintf("max|dW|=%.3g iter %s vs %s", max(abs(a@w - dn@w)), a@misc$iter, dn@misc$iter))
  }
  # unconstrained ALS must reach the Eckart-Young optimum
  S <- sim(200, 150, 5, noise = 0.3, seed = 9); sv <- base::svd(S$A); opt <- sum(sv$d[-(1:5)]^2) / length(S$A)
  for (nn in list(c(FALSE, FALSE), c(FALSE, TRUE), c(TRUE, FALSE), c(TRUE, TRUE))) for (s in c("cd", "cholesky")) {
    r <- fit(S$A, 5, seed = 2, nonneg = nn, tol = 1e-8, maxit = 300, solver = s); lab <- paste0("nonneg=", paste(nn, collapse = ","), " ", s)
    if (is_err(r)) { rec("nonneg_flags", lab, "", NA, "ERR", r); next }
    e <- mse_of(r, S$A)
    st <- if (!nn[1] && !nn[2]) (if (abs(e - opt) / opt < 1e-3) "PASS" else "FAIL") else if (e >= opt * (1 - 1e-6)) "PASS" else "FAIL"
    rec("nonneg_flags", lab, "mse / svd optimum", e / opt, st, sprintf("min(w)=%.3g min(h)=%.3g", min(r@w), min(r@h)))
    if (nn[1] && min(r@w) < 0) rec("nonneg_violation", lab, "min(w)", min(r@w), "FAIL")
    if (nn[2] && min(r@h) < 0) rec("nonneg_violation", lab, "min(h)", min(r@h), "FAIL")
  }
  # norm options and scaling invariants
  for (nm in c("L1", "L2", "none")) { r <- fit(S$A, 5, seed = 2, norm = nm); if (is_err(r)) { rec("norm", nm, "", NA, "ERR", r); next }
    cs <- if (nm == "L1") colSums(r@w) else sqrt(colSums(r@w^2)); rs <- if (nm == "L1") rowSums(r@h) else sqrt(rowSums(r@h^2))
    rec("norm_scaling", nm, "range of factor norms", max(abs(c(cs, rs) - 1)), if (nm == "none" || max(abs(c(cs, rs) - 1)) < 1e-4) "PASS" else "FAIL", sprintf("w norms %.3g-%.3g h norms %.3g-%.3g d sorted=%s", min(cs), max(cs), min(rs), max(rs), !is.unsorted(rev(r@d)))) }
  # nnls() against the reference
  set.seed(4); W <- matrix(runif(60 * 6), 60, 6); Ht <- matrix(rexp(6 * 30) * (runif(180) < 0.5), 6, 30); A <- W %*% Ht + 0.05 * matrix(rnorm(60 * 30), 60, 30)
  G <- crossprod(W); B <- crossprod(W, A); Href <- sapply(1:30, function(j) nnls_ref(G, B[, j]))
  h <- try_(nnls(w = W, A = A)); if (is_err(h)) rec("nnls_plain", "nnls()", "", NA, "ERR", h) else rec("nnls_plain", "nnls()", "max|h-ref|", max(abs(h - Href)), if (max(abs(h - Href)) < 1e-4) "PASS" else "FAIL")
  for (lam in c(0.05, 0.3)) { h <- try_(nnls(w = W, A = A, L1 = c(0, lam))); if (is_err(h)) { rec("nnls_L1_semantics", "nnls()", lam, NA, "ERR", h); next }
    cand <- list(absolute = lam, "x max(b)" = lam * max(B), "x mean diag(G)" = lam * mean(diag(G)), "x max diag(G)" = lam * max(diag(G)))
    d <- sapply(cand, function(l) max(abs(h - sapply(1:30, function(j) nnls_ref(G, B[, j], l1 = l)))))
    rec("nnls_L1_semantics", "nnls()", lam, min(d), if (min(d) < 1e-3) "INFO" else "FAIL", sprintf("closest scaling: %s (max diff %.2e); others %s", names(d)[which.min(d)], min(d), paste(names(d), signif(d, 2), collapse = ", "))) }
  for (lam in c(0.1, 1)) { h <- try_(nnls(w = W, A = A, L2 = c(0, lam))); if (is_err(h)) { rec("nnls_L2_semantics", "nnls()", lam, NA, "ERR", h); next }
    cand <- list(absolute = lam, "x mean diag(G)" = lam * mean(diag(G)), "x max diag(G)" = lam * max(diag(G)))
    d <- sapply(cand, function(l) max(abs(h - sapply(1:30, function(j) nnls_ref(G + diag(l, 6), B[, j])))))
    rec("nnls_L2_semantics", "nnls()", lam, min(d), if (min(d) < 1e-3) "INFO" else "FAIL", sprintf("closest: %s; all %s", names(d)[which.min(d)], paste(names(d), signif(d, 2), collapse = ", "))) }
  ub <- 0.5 * max(Href); h <- try_(nnls(w = W, A = A, upper_bound = c(0, ub)))
  if (is_err(h)) rec("nnls_upper_bound", "nnls()", ub, NA, "ERR", h) else { hb <- sapply(1:30, function(j) nnls_ref(G, B[, j], upper = ub))
    rec("nnls_upper_bound", "nnls()", "max|h-boxref|", max(abs(h - hb)), if (max(abs(h - hb)) < 1e-4) "PASS" else "FAIL", sprintf("max(h)=%.4g bound=%.4g; vs clipped-NNLS diff=%.2e", max(h), ub, max(abs(h - pmin(Href, ub))))) }
  h <- try_(nnls(w = W, A = A, nonneg = c(TRUE, FALSE))); if (!is_err(h)) rec("nnls_unconstrained", "nnls()", "max|h-OLS|", max(abs(h - solve(G, B))), if (max(abs(h - solve(G, B))) < 1e-3) "PASS" else "FAIL") else rec("nnls_unconstrained", "nnls()", "", NA, "ERR", h)
  # predict / evaluate / mse helpers
  r <- fit(S$A, 5, seed = 2, solver = "cd"); p <- try_(predict(r, S$A)); if (is_err(p)) rec("predict", "nmf", "", NA, "ERR", p) else { ph <- if (is.matrix(p)) p else if (isS4(p)) p@h else p$h
    rec("predict_recovers_h", "nmf", "cor(pred, d*h)", cor(as.vector(ph), as.vector(r@d * r@h)), "INFO", sprintf("dim %s; scale ratio %.3g", paste(dim(ph), collapse = "x"), mean(ph) / mean(r@d * r@h))) }
  ev <- try_(evaluate(r, S$A)); if (is_err(ev)) rec("evaluate", "nmf", "", NA, "ERR", ev) else rec("evaluate_vs_explicit", "nmf", "rel diff", abs(ev - mse_of(r, S$A)) / mse_of(r, S$A), if (abs(ev - mse_of(r, S$A)) / mse_of(r, S$A) < 1e-3) "PASS" else "FAIL")
}

# =========================================================================================
if (SECTION == "B") {  # every penalty on every fit path: does it act, and in the right direction
  S <- sim(240, 160, 6, noise = 0.2, seed = 21, sparse_frac = 0.3); Ad <- S$A; As <- as(Ad, "CsparseMatrix"); Ac <- as(round(Ad * 5), "CsparseMatrix"); Acd <- as.matrix(Ac)
  GW <- chain(240); GH <- chain(160)
  set.seed(5); Mk <- as(matrix(runif(240 * 160) < 0.1, 240, 160) * 1, "CsparseMatrix")
  spz <- tempfile(fileext = ".spz"); sp_ok <- !is_err(try_(st_write(As, spz)))
  Sy <- { B <- sim(160, 12, 6, noise = 0.05, seed = 2)$A; tcrossprod(B) }
  lab <- rep(1:4, each = 40)
  paths <- list(
    "sparse mse cd"       = list(A = As, a = list(solver = "cd")),
    "sparse mse cholesky" = list(A = As, a = list(solver = "cholesky")),
    "dense mse cd"        = list(A = Ad, a = list(solver = "cd")),
    "dense mse cholesky"  = list(A = Ad, a = list(solver = "cholesky")),
    "sparse gp"           = list(A = Ac, a = list(loss = "gp")),
    "dense gp"            = list(A = Acd, a = list(loss = "gp")),
    "sparse nb"           = list(A = Ac, a = list(loss = "nb")),
    "sparse gamma"        = list(A = As, a = list(loss = "gamma")),
    "sparse tweedie"      = list(A = Ac, a = list(loss = "tweedie")),
    "robust huber"        = list(A = As, a = list(robust = TRUE)),
    "mask zeros"          = list(A = As, a = list(mask = "zeros")),
    "custom mask"         = list(A = As, a = list(mask = Mk)),
    "cv test_fraction"    = list(A = As, a = list(test_fraction = 0.1, cv_seed = 3)),
    "cv gp"               = list(A = Ac, a = list(test_fraction = 0.1, cv_seed = 3, loss = "gp")),
    "zi row gp"           = list(A = Ac, a = list(loss = "gp", zi = "row")),
    "projective"          = list(A = As, a = list(projective = TRUE)),
    "symmetric"           = list(A = as(Sy, "CsparseMatrix"), a = list(symmetric = TRUE)),
    "nonneg FF"           = list(A = As, a = list(nonneg = c(FALSE, FALSE))),
    "streaming spz"       = if (sp_ok) list(A = spz, a = list()) else NULL)
  paths <- paths[!vapply(paths, is.null, TRUE)]
  if (RES == "gpu") paths <- paths[c("sparse mse cd", "sparse mse cholesky", "dense mse cd", "sparse gp", "dense gp", "sparse nb", "robust huber", "mask zeros", "custom mask", "cv test_fraction", "cv gp", "zi row gp", "projective", "symmetric")]
  sparsity <- function(M) mean(M == 0); zrows <- function(M) mean(rowSums(abs(M)) == 0)
  for (pn in names(paths)) { P <- paths[[pn]]; nr <- if (pn == "symmetric") 160 else 240; ncn <- 160
    f <- function(...) do.call(fit, c(list(P$A, 6, seed = 4, tol = 1e-5, maxit = 40), P$a, list(...)))
    b0 <- f(); if (is_err(b0)) { rec("baseline", pn, "", NA, "ERR", b0); next }
    if (!isS4(b0)) { rec("baseline", pn, "", NA, "ERR", paste("returned", class(b0)[1])); next }
    rec("baseline", pn, "iter", b0@misc$iter, "INFO", sprintf("loss=%.5g", if (is.null(b0@misc$loss)) NA else b0@misc$loss))
    gw <- if (pn == "symmetric") chain(160) else GW
    tgt <- { Tm <- try_(compute_target(b0@h, lab)); if (is_err(Tm)) NULL else Tm }
    tests <- list(
      list("L1 on W",  function(l) f(L1 = c(l, 0)), c(0.05, 0.3, 0.8), function(r) sparsity(r@w), +1, "sparsity(W)"),
      list("L1 on H",  function(l) f(L1 = c(0, l)), c(0.05, 0.3, 0.8), function(r) sparsity(r@h), +1, "sparsity(H)"),
      list("L2 on W",  function(l) f(L2 = c(l, 0)), c(0.1, 1, 10), function(r) cosmean(r@w), 0, "mean|cor| W cols"),
      list("L21 on W", function(l) f(L21 = c(l, 0)), c(0.05, 0.3, 0.8), function(r) zrows(r@w), +1, "zero rows of W"),
      list("angular W", function(l) f(angular = c(l, 0)), c(0.1, 0.5, 2), function(r) cosmean(r@w), -1, "mean|cor| W cols"),
      list("angular H", function(l) f(angular = c(0, l)), c(0.1, 0.5, 2), function(r) cosmean(t(r@h)), -1, "mean|cor| H rows"),
      list("graph_W",  function(l) f(graph_W = lapl(gw), graph_lambda = c(l, 0)), c(0.5, 5, 50), function(r) lap_energy(t(r@w), gw), -1, "Laplacian energy W"),
      list("graph_H",  function(l) f(graph_H = lapl(GH), graph_lambda = c(0, l)), c(0.5, 5, 50), function(r) lap_energy(r@h, GH), -1, "Laplacian energy H"),
      list("upper_bound W", function(l) f(upper_bound = c(l * max(b0@w), 0)), c(0.8, 0.5, 0.2), function(r) max(r@w) / max(b0@w), -1, "max(W)/max(W0)"),
      list("upper_bound H", function(l) f(upper_bound = c(0, l * max(b0@h))), c(0.8, 0.5, 0.2), function(r) max(r@h) / max(b0@h), -1, "max(H)/max(H0)"),
      if (!is.null(tgt)) list("target_H", function(l) f(target_H = tgt, target_lambda = l), c(0.05, 0.2, 0.5), function(r) { Tr <- try_(compute_target(r@h, lab)); cor(as.vector(r@h), as.vector(tgt)) }, +1, "cor(H, target)"))
    for (T in tests) { if (is.null(T)) next
      rs <- lapply(T[[3]], T[[2]]); er <- vapply(rs, is_err, TRUE)
      if (all(er)) { rec("penalty", pn, T[[1]], NA, "REJECTED", rs[[1]]); next }
      rs <- rs[!er]; if (!all(vapply(rs, isS4, TRUE))) { rec("penalty", pn, T[[1]], NA, "ERR", "non-S4 result"); next }
      dW <- max(vapply(rs, function(r) max(abs(r@w - b0@w), abs(r@h - b0@h)), 0))
      st0 <- T[[4]](b0); st <- vapply(rs, T[[4]], 0); dirn <- T[[5]]
      moved <- dW > 1e-6
      ok_dir <- if (dirn == 0) TRUE else (dirn * (st[length(st)] - st0) > 0) && all(dirn * diff(c(st0, st)) >= -0.02 * max(abs(c(st0, st)), 1e-12))
      bound_ok <- TRUE; if (grepl("upper_bound", T[[1]])) { bound_ok <- all(st <= T[[3]][seq_along(st)] * 1.001); ok_dir <- bound_ok }
      status <- if (!isTRUE(moved)) "NOOP" else if (!isTRUE(ok_dir)) "WRONGDIR" else "PASS"
      rec("penalty", pn, T[[1]], dW, status, sprintf("%s: base %.4g -> %s (lambda %s)", T[[6]], st0, paste(signif(st, 4), collapse = ", "), paste(T[[3]][!er], collapse = ", ")))
    }
  }
}

# =========================================================================================
if (SECTION == "C") {  # distributions: does each loss minimize its own deviance
  dev <- list(
    pois = function(y, mu) { mu <- pmax(mu, 1e-10); 2 * mean(ifelse(y > 0, y * log(y / mu), 0) - (y - mu)) },
    nb = function(y, mu, r = 5) { mu <- pmax(mu, 1e-10); 2 * mean(ifelse(y > 0, y * log(y / mu), 0) - (y + r) * log((y + r) / (mu + r))) },
    gamma = function(y, mu) { mu <- pmax(mu, 1e-10); 2 * mean((y - mu) / mu - log(y / mu)) },
    ig = function(y, mu) { mu <- pmax(mu, 1e-10); mean((y - mu)^2 / (mu^2 * y)) },
    tw = function(y, mu, p = 1.5) { mu <- pmax(mu, 1e-10); 2 * mean(y^(2 - p) / ((1 - p) * (2 - p)) - y * mu^(1 - p) / (1 - p) + mu^(2 - p) / (2 - p)) })
  set.seed(31); m <- 300; n <- 200; k <- 5
  W <- matrix(rexp(m * k) * (runif(m * k) < 0.5), m, k) + 0.02; H <- matrix(rexp(k * n) * (runif(k * n) < 0.6), k, n) + 0.02
  rinvg <- function(mu, lam) { nn <- length(mu); y <- rnorm(nn)^2; x <- mu + mu^2 * y / (2 * lam) - mu / (2 * lam) * sqrt(4 * mu * lam * y + mu^2 * y^2); ifelse(runif(nn) <= mu / (mu + x), x, mu^2 / x) }
  rtw <- function(mu, p = 1.5, phi = 1) { lam <- mu^(2 - p) / (phi * (2 - p)); al <- (2 - p) / (p - 1); ga <- phi * (p - 1) * mu^(p - 1); N <- rpois(length(mu), lam); ifelse(N > 0, rgamma(length(mu), shape = pmax(N, 1e-9) * al, scale = ga), 0) }
  data <- list(
    poisson_low  = list(mu = 0.5 * W %*% H, gen = function(mu) rpois(length(mu), mu), loss = "gp", extra = list(dispersion = "none"), dev = dev$pois),
    poisson_high = list(mu = 8 * W %*% H, gen = function(mu) rpois(length(mu), mu), loss = "gp", extra = list(dispersion = "none"), dev = dev$pois),
    gp_default   = list(mu = 2 * W %*% H, gen = function(mu) rpois(length(mu), mu), loss = "gp", extra = list(), dev = dev$pois),
    negbin_r5    = list(mu = 3 * W %*% H, gen = function(mu) rnbinom(length(mu), size = 5, mu = mu), loss = "nb", extra = list(), dev = dev$nb),
    gamma_shape4 = list(mu = 3 * W %*% H + 0.2, gen = function(mu) rgamma(length(mu), shape = 4, scale = mu / 4), loss = "gamma", extra = list(), dev = dev$gamma),
    invgauss     = list(mu = 3 * W %*% H + 0.5, gen = function(mu) rinvg(mu, 4), loss = "inverse_gaussian", extra = list(), dev = dev$ig),
    tweedie_1.5  = list(mu = 2 * W %*% H, gen = function(mu) rtw(mu), loss = "tweedie", extra = list(), dev = dev$tw))
  recov <- function(r, Wt) { C <- abs(cor(r@w, Wt)); mean(apply(C, 2, max)) }
  for (dn in names(data)) { D <- data[[dn]]; Y <- matrix(D$gen(as.vector(D$mu)), m, n); Ys <- as(Y, "CsparseMatrix")
    oracle <- D$dev(Y, D$mu)
    for (inp in c("sparse", "dense")) { A <- if (inp == "sparse") Ys else Y; path <- paste(dn, inp)
      ff <- function(loss, ...) do.call(fit, c(list(A, k, seed = 3, tol = 1e-6, maxit = 100, loss = loss), list(...)))
      r0 <- ff("mse"); r1 <- do.call(ff, c(list(D$loss), D$extra))
      if (is_err(r1)) { rec("matching_loss", path, D$loss, NA, "ERR", r1); next }
      d0 <- D$dev(Y, recon(r0)); d1 <- D$dev(Y, recon(r1))
      rec("deviance_vs_mse_fit", path, D$loss, d1 / d0, if (d1 <= d0 * 1.001) "PASS" else "FAIL", sprintf("true-distribution deviance: matching loss %.5g, mse fit %.5g, at true mu %.5g; iter %s/%s", d1, d0, oracle, r1@misc$iter, r0@misc$iter))
      rec("factor_recovery", path, D$loss, recov(r1, W) - recov(r0, W), "INFO", sprintf("mean max|cor| with true W: matching %.4f, mse %.4f", recov(r1, W), recov(r0, W)))
      tr <- sapply(c(1, 2, 3, 5, 8, 12, 20, 40), function(it) { r <- do.call(ff, c(list(D$loss, maxit = it, tol = 0), D$extra)); if (is_err(r)) NA else D$dev(Y, recon(r)) })
      if (!anyNA(tr)) rec("deviance_monotone", path, D$loss, max(diff(tr)) / abs(tr[length(tr)]), if (max(diff(tr)) <= 1e-4 * abs(tr[length(tr)])) "PASS" else "FAIL", paste(signif(tr, 5), collapse = " "))
      if (!is.null(r1@misc$loss)) rec("reported_loss", path, D$loss, r1@misc$loss, "INFO", sprintf("reported %.6g; own deviance %.6g; ratio %.4g", r1@misc$loss, d1, r1@misc$loss / d1))
      for (nm in intersect(names(r1@misc), c("theta", "dispersion", "pi_row", "pi_col"))) rec("dispersion_estimate", path, nm, mean(r1@misc[[nm]]), "INFO", sprintf("range %.4g-%.4g", min(r1@misc[[nm]]), max(r1@misc[[nm]])))
      # same data, CV path with a small holdout must optimize the same objective
      rc <- do.call(ff, c(list(D$loss, test_fraction = 0.02, cv_seed = 1), D$extra))
      if (is_err(rc) || !isS4(rc)) rec("cv_path_same_objective", path, D$loss, NA, "ERR", if (is_err(rc)) rc else class(rc)[1]) else { dc <- D$dev(Y, recon(rc))
        rec("cv_path_same_objective", path, D$loss, dc / d1, if (abs(dc / d1 - 1) < 0.05) "PASS" else "FAIL", sprintf("deviance cv-fit %.5g vs plain fit %.5g (2%% held out)", dc, d1)) }
      # mask + non-Gaussian loss: must differ from mask + mse
      set.seed(8); Mk <- as(matrix(runif(m * n) < 0.1, m, n) * 1, "CsparseMatrix")
      rm1 <- do.call(ff, c(list(D$loss, mask = Mk), D$extra)); rm0 <- ff("mse", mask = Mk)
      if (is_err(rm1) || is_err(rm0)) rec("mask_with_loss", path, D$loss, NA, "ERR", if (is_err(rm1)) rm1 else rm0) else rec("mask_with_loss", path, D$loss, max(abs(rm1@w - rm0@w)), if (max(abs(rm1@w - rm0@w)) > 1e-6) "PASS" else "NOOP", "max|W(mask+loss) - W(mask+mse)|; zero means the loss is ignored when a mask is set")
    }
    rs <- do.call(fit, c(list(Ys, k, seed = 3, tol = 1e-6, maxit = 100, loss = D$loss), D$extra)); rd <- do.call(fit, c(list(Y, k, seed = 3, tol = 1e-6, maxit = 100, loss = D$loss), D$extra))
    if (!is_err(rs) && !is_err(rd)) rec("sparse_vs_dense_same_fit", dn, D$loss, abs(D$dev(Y, recon(rs)) / D$dev(Y, recon(rd)) - 1), if (abs(D$dev(Y, recon(rs)) / D$dev(Y, recon(rd)) - 1) < 0.01) "PASS" else "FAIL", sprintf("deviance sparse-input %.5g vs dense-input %.5g; max|dW|=%.3g", D$dev(Y, recon(rs)), D$dev(Y, recon(rd)), max(abs(rs@w - rd@w))))
  }
  # zero inflation: structural zeros by row
  set.seed(77); mu <- 4 * W %*% H; pi_row <- runif(m, 0, 0.6); Y <- matrix(rpois(m * n, mu), m, n); Y[matrix(runif(m * n) < pi_row, m, n)] <- 0; Ys <- as(Y, "CsparseMatrix")
  for (ls in c("gp", "nb")) { r0 <- fit(Ys, k, seed = 3, loss = ls); r1 <- fit(Ys, k, seed = 3, loss = ls, zi = "row")
    if (is_err(r1)) { rec("zero_inflation", ls, "zi=row", NA, "ERR", r1); next }
    e0 <- mean(abs(recon(r0) - mu)) ; e1 <- mean(abs(recon(r1) - mu))
    rec("zi_recovers_mean", ls, "zi=row", e1 / e0, if (e1 < e0) "PASS" else "FAIL", sprintf("mean|mu_hat - mu_true|: zi %.4g vs no-zi %.4g", e1, e0))
    if (!is.null(r1@misc$pi_row)) rec("zi_pi_estimate", ls, "cor(pi_hat, pi_true)", cor(r1@misc$pi_row, pi_row), if (cor(r1@misc$pi_row, pi_row) > 0.8) "PASS" else "FAIL", sprintf("mean pi_hat %.3f vs true %.3f", mean(r1@misc$pi_row), mean(pi_row))) }
  # robust: gross outliers
  set.seed(78); S <- sim(300, 200, 5, noise = 0.05, seed = 6); Y <- S$A; idx <- sample(length(Y), 0.02 * length(Y)); Yo <- Y; Yo[idx] <- Yo[idx] + 20 * max(Y)
  for (rb in list(FALSE, TRUE, "mae", 0.5)) { r <- fit(as(Yo, "CsparseMatrix"), 5, seed = 3, robust = rb, maxit = 100); lab <- paste0("robust=", rb)
    if (is_err(r)) { rec("robust_outliers", lab, "", NA, "ERR", r); next }
    rec("robust_outliers", lab, "mse to clean data", mean((recon(r) - Y)^2), "INFO", sprintf("iter %s", r@misc$iter)) }
  # input validation
  bad <- list("negative data + gp" = function() fit(-abs(S$A), 3, loss = "gp"), "zeros + gamma (dense)" = function() fit(S$A * (S$A > 0.5), 3, loss = "gamma"), "non-integer + nb" = function() fit(S$A, 3, loss = "nb"), "tweedie_power=0.5" = function() fit(S$A, 3, loss = "tweedie", tweedie_power = 0.5), "tweedie_power=1" = function() fit(round(S$A), 3, loss = "tweedie", tweedie_power = 1), "tweedie_power=2" = function() fit(S$A + 0.1, 3, loss = "tweedie", tweedie_power = 2), "NA in data" = function() { X <- S$A; X[1:5, 1] <- NA; fit(X, 3) }, "Inf in data" = function() { X <- S$A; X[1, 1] <- Inf; fit(X, 3) }, "k > min(m,n)" = function() fit(S$A[1:10, 1:8], 9), "negative L1" = function() fit(S$A, 3, L1 = -0.1), "L1 > 1" = function() fit(S$A, 3, L1 = 1.5), "target_H wrong dim" = function() fit(S$A, 3, target_H = matrix(1, 2, 5), target_lambda = 0.1), "graph_W wrong dim" = function() fit(S$A, 3, graph_W = chain(7), graph_lambda = 1), "all-zero column" = function() { X <- S$A; X[, 3] <- 0; fit(X, 3) }, "all-zero matrix" = function() fit(matrix(0, 20, 10), 2))
  for (bn in names(bad)) { r <- bad[[bn]](); rec("input_validation", bn, "", NA, if (is_err(r)) "REJECTED" else if (any(!is.finite(r@w)) || any(!is.finite(r@h))) "FAIL" else "ACCEPTED", if (is_err(r)) r else sprintf("finite result; loss=%s", format(r@misc$loss))) }
}

# =========================================================================================
if (SECTION == "D") {  # cross-validation and rank selection
  mk <- function(noise, seed, count = FALSE) { set.seed(seed); m <- 400; n <- 300; k <- 5
    W <- matrix(rexp(m * k) * (runif(m * k) < 0.4), m, k); H <- matrix(rexp(k * n) * (runif(k * n) < 0.5), k, n); mu <- W %*% H
    if (count) matrix(rpois(m * n, 3 * mu), m, n) else pmax(mu + noise * mean(mu) * matrix(rnorm(m * n), m, n), 0) }
  Ad <- mk(0.5, 1); As <- as(Ad, "CsparseMatrix"); Ac <- as(mk(0, 2, TRUE), "CsparseMatrix")
  cvrun <- function(A, ...) try_(suppressWarnings(as.data.frame(RcppML::nmf(A, k = 2:9, seed = 1, resource = RES, ...))))
  best <- function(df) tapply(seq_len(nrow(df)), df$rep, function(ix) df$k[ix][which.min(df$test_mse[ix])])
  cases <- list("dense mse" = list(Ad), "sparse mse" = list(As), "sparse mse cd" = list(As, solver = "cd"), "sparse mse cholesky" = list(As, solver = "cholesky"), "sparse mask=zeros" = list(As, mask = "zeros"), "counts gp" = list(Ac, loss = "gp"), "counts nb" = list(Ac, loss = "nb"), "counts mse" = list(Ac), "sparse L1=0.1" = list(As, L1 = c(0.1, 0.1)), "sparse maxit=30 patience=3" = list(As, maxit = 30, patience = 3))
  for (cn in names(cases)) { df <- do.call(cvrun, c(cases[[cn]], list(test_fraction = 0.1, cv_seed = 1:3)))
    if (is_err(df)) { rec("rank_recovery", cn, "true k=5", NA, "ERR", df); next }
    b <- best(df); mm <- tapply(df$test_mse, df$k, mean)
    rec("rank_recovery", cn, "true k=5", mean(b), if (all(b == 5)) "PASS" else if (any(b == 5)) "PARTIAL" else "FAIL", sprintf("argmin k per replicate: %s; mean test_mse by k: %s", paste(b, collapse = ","), paste(names(mm), signif(mm, 4), sep = ":", collapse = " ")))
    rec("train_below_test_at_high_k", cn, "k=9", mean(df$train_mse[df$k == 9]) / mean(df$test_mse[df$k == 9]), "INFO", sprintf("train %.4g test %.4g; best_iter>total_iter in %d of %d rows", mean(df$train_mse[df$k == 9]), mean(df$test_mse[df$k == 9]), sum(df$best_iter > df$total_iter), nrow(df))) }
  # holdout fraction granularity: fractions that floor to the same 1/f give identical masks
  one <- function(f, ...) { r <- try_(suppressWarnings(RcppML::nmf(As, 5, seed = 1, test_fraction = f, cv_seed = 9, resource = RES, maxit = 30, ...))); if (is_err(r)) NA else if (isS4(r)) r@misc$test_loss else NA }
  fr <- c(0.05, 0.1, 0.2, 0.21, 0.25, 0.26, 0.3, 0.33, 0.34, 0.4, 0.5, 0.51, 0.6, 0.75, 0.9); tl <- sapply(fr, one)
  rec("holdout_fraction_grid", "sparse mse", "test loss by fraction", NA, "INFO", paste(fr, signif(tl, 5), sep = ":", collapse = " "))
  same <- function(a, b) isTRUE(all.equal(tl[fr == a], tl[fr == b], tolerance = 1e-10))
  rec("holdout_fraction_distinct", "sparse mse", "0.21 vs 0.25", NA, if (same(0.21, 0.25)) "FAIL" else "PASS", "identical test loss means both requests held out the same entries")
  rec("holdout_fraction_distinct", "sparse mse", "0.26 vs 0.33", NA, if (same(0.26, 0.33)) "FAIL" else "PASS", "")
  rec("holdout_fraction_distinct", "sparse mse", "0.34 vs 0.5", NA, if (same(0.34, 0.5)) "FAIL" else "PASS", "")
  rec("holdout_fraction_above_half", "sparse mse", "0.6 / 0.5 test loss", tl[fr == 0.6] / tl[fr == 0.5], if (is.finite(tl[fr == 0.6]) && tl[fr == 0.6] < 3 * tl[fr == 0.5]) "PASS" else "FAIL", "a jump means every entry was held out")
  # a small holdout must reproduce the plain fit
  for (sv in c("cd", "cholesky")) { r0 <- fit(As, 5, seed = 1, solver = sv, tol = 1e-6, maxit = 80); r1 <- fit(As, 5, seed = 1, solver = sv, tol = 1e-6, maxit = 80, test_fraction = 0.01, cv_seed = 2, patience = 1000)
    if (is_err(r1) || !isS4(r1)) { rec("cv_small_holdout_matches_plain", sv, "", NA, "ERR", if (is_err(r1)) r1 else class(r1)[1]); next }
    rec("cv_small_holdout_matches_plain", sv, "mse ratio on all entries", mse_of(r1, Ad) / mse_of(r0, Ad), if (abs(mse_of(r1, Ad) / mse_of(r0, Ad) - 1) < 0.02) "PASS" else "FAIL", sprintf("reported train %.5g test %.5g; explicit all-entry mse %.5g; iter %s vs %s", r1@misc$train_loss, r1@misc$test_loss, mse_of(r1, Ad), r1@misc$iter, r0@misc$iter))
    gh <- gap_H(r1@w %*% diag(r1@d), r1@h, Ad)["mean"]; rec("cv_halfstep_gap", sv, "H|W", gh, "INFO", "optimality gap on all entries (expected small but nonzero because 1% is held out)") }
  # reproducibility and seed independence
  a <- cvrun(As, test_fraction = 0.1, cv_seed = 5); b <- cvrun(As, test_fraction = 0.1, cv_seed = 5); c2 <- cvrun(As, test_fraction = 0.1, cv_seed = 6)
  if (!is_err(a) && !is_err(c2)) { rec("cv_reproducible", "sparse mse", "same cv_seed", max(abs(a$test_mse - b$test_mse)), if (max(abs(a$test_mse - b$test_mse)) == 0) "PASS" else "FAIL"); rec("cv_seed_changes_mask", "sparse mse", "different cv_seed", max(abs(a$test_mse - c2$test_mse)), if (max(abs(a$test_mse - c2$test_mse)) > 0) "PASS" else "FAIL") }
  # automatic rank
  for (cn in c("dense mse", "sparse mse", "counts gp")) { r <- try_(suppressWarnings(do.call(RcppML::nmf, c(list(cases[[cn]][[1]], k = "auto", seed = 1, resource = RES), cases[[cn]][-1])))); if (is_err(r)) { rec("auto_rank", cn, "true k=5", NA, "ERR", r); next }
    kk <- ncol(r@w); rec("auto_rank", cn, "true k=5", kk, if (kk == 5) "PASS" else if (abs(kk - 5) <= 1) "PARTIAL" else "FAIL", paste(capture.output(str(r@misc$rank_search, give.attr = FALSE)), collapse = " ")) }
  # track_train_loss
  df <- cvrun(As, test_fraction = 0.1, cv_seed = 1, track_train_loss = FALSE); if (!is_err(df)) rec("track_train_loss_FALSE", "sparse mse", "zeros in train_mse", sum(df$train_mse == 0), if (all(df$train_mse == 0 | is.na(df$train_mse)) || all(df$train_mse > 0)) "PASS" else "FAIL", paste(signif(df$train_mse, 3), collapse = " "))
  # SVD cross-validation
  for (meth in c("auto", "deflation", "krylov")) for (cen in c(FALSE, TRUE)) { r <- try_(suppressWarnings(RcppML::svd(Ad, k = "auto", test_fraction = 0.1, seed = 1, method = meth, center = cen, resource = RES))); lab <- sprintf("svd %s center=%s", meth, cen)
    if (is_err(r)) { rec("svd_auto_rank", lab, "true k=5", NA, "ERR", r); next }
    kk <- length(r@d); rec("svd_auto_rank", lab, "true k=5", kk, if (kk == 5) "PASS" else if (abs(kk - 5) <= 1) "PARTIAL" else "FAIL", sprintf("misc: %s", paste(names(r@misc), collapse = ","))) }
  r <- try_(suppressWarnings(RcppML::svd(Ad, k = 8, test_fraction = 0.1, seed = 1, resource = RES))); if (!is_err(r)) { sv <- base::svd(Ad)$d[1:8]
    rec("svd_cv_singular_value_bias", "svd k=8 test_fraction=0.1", "d / true d", mean(r@d[1:5] / sv[1:5]), "INFO", sprintf("ratio per component: %s (0.9 would mean held-out entries are zero-filled)", paste(signif(r@d / sv[seq_along(r@d)], 4), collapse = " "))) }
}

# =========================================================================================
if (SECTION == "E") {  # SVD and PCA against base::svd and prcomp
  set.seed(41); mats <- list(
    "dense 200x100 decaying" = { U <- qr.Q(qr(matrix(rnorm(200 * 100), 200))); V <- qr.Q(qr(matrix(rnorm(100 * 100), 100))); U %*% (0.85^(0:99) * 100 * t(V)) },
    "dense wide 60x300" = matrix(rnorm(60 * 300), 60) %*% diag(seq(3, 0.1, length.out = 300)),
    "sparse 500x300 5pct" = rsparsematrix(500, 300, 0.05),
    "exact rank 5 (30x20)" = { U <- qr.Q(qr(matrix(rnorm(150), 30))); V <- qr.Q(qr(matrix(rnorm(100), 20))); U %*% (c(50, 30, 15, 8, 4) * t(V)) },
    "repeated singular values" = { U <- qr.Q(qr(matrix(rnorm(80 * 40), 80))); V <- qr.Q(qr(matrix(rnorm(40 * 40), 40))); U %*% (c(10, 10, 10, 5, 5, rep(1, 35)) * t(V)) },
    "nonneg counts 300x200" = as(matrix(rpois(300 * 200, 0.3), 300), "CsparseMatrix"))
  methods <- c("auto", "deflation", "krylov", "lanczos", "irlba", "randomized")
  for (mn in names(mats)) { A <- mats[[mn]]; Ad <- as.matrix(A); ref <- base::svd(Ad); kmax <- min(dim(Ad))
    for (k in c(1, 5, 20)) { if (k > kmax) next
      for (meth in methods) { path <- sprintf("%s | %s k=%d", mn, meth, k)
        r <- try_(suppressWarnings(RcppML::svd(A, k = k, method = meth, seed = 1, resource = RES)))
        if (is_err(r)) { rec("svd_fit", path, "", NA, "ERR", r); next }
        kk <- length(r@d); dref <- ref$d[seq_len(kk)]; rel <- max(abs(r@d - dref) / dref[1])
        ou <- max(abs(crossprod(r@u) - diag(kk))); ov <- max(abs(crossprod(r@v) - diag(kk)))
        Rk <- r@u %*% (r@d * t(r@v)); opt <- sum(ref$d[-seq_len(kk)]^2); err <- sum((Ad - Rk)^2); exc <- (err - opt) / sum(ref$d^2)
        rec("svd_values_vs_base", path, "max|d-d_ref|/d1", rel, if (rel < 1e-3) "PASS" else "FAIL", sprintf("d: %s | ref: %s", paste(signif(r@d[1:min(kk, 5)], 5), collapse = " "), paste(signif(dref[1:min(kk, 5)], 5), collapse = " ")))
        rec("svd_orthogonality", path, "max|UtU-I|,|VtV-I|", max(ou, ov), if (max(ou, ov) < 1e-3) "PASS" else "FAIL", sprintf("U %.2e V %.2e", ou, ov))
        rec("svd_recon_excess", path, "(err-opt)/||A||^2", exc, if (exc < 1e-4) "PASS" else "FAIL", sprintf("k returned %d", kk))
      } } }
  # PCA against prcomp
  X <- mats[["dense 200x100 decaying"]] + matrix(rep(rnorm(100, 5), each = 200), 200); pr <- prcomp(X, center = TRUE, scale. = FALSE); pr2 <- prcomp(X, center = TRUE, scale. = TRUE)
  for (meth in methods) { r <- try_(suppressWarnings(RcppML::pca(X, k = 5, method = meth, seed = 1, resource = RES))); if (is_err(r)) { rec("pca_vs_prcomp", meth, "center", NA, "ERR", r); next }
    cand <- c("d = sdev*sqrt(n-1) (cols are variables)" = max(abs(r@d - pr$sdev[1:5] * sqrt(199)) / (pr$sdev[1] * sqrt(199))), "rows are variables" = { p2 <- prcomp(t(X)); max(abs(r@d - p2$sdev[1:5] * sqrt(99)) / (p2$sdev[1] * sqrt(99))) })
    rec("pca_vs_prcomp", meth, "center", min(cand), if (min(cand) < 1e-3) "PASS" else "FAIL", sprintf("matches: %s; all: %s", names(cand)[which.min(cand)], paste(signif(cand, 3), collapse = ", "))) }
  for (meth in c("auto", "deflation", "irlba")) { r <- try_(suppressWarnings(RcppML::svd(X, k = 5, center = TRUE, scale = TRUE, method = meth, seed = 1, resource = RES))); if (is_err(r)) { rec("pca_scaled_vs_prcomp", meth, "center+scale", NA, "ERR", r); next }
    cand <- c(cols = max(abs(r@d - pr2$sdev[1:5] * sqrt(199)) / (pr2$sdev[1] * sqrt(199))), rows = { p2 <- prcomp(t(X), scale. = TRUE); max(abs(r@d - p2$sdev[1:5] * sqrt(99)) / (p2$sdev[1] * sqrt(99))) })
    rec("pca_scaled_vs_prcomp", meth, "center+scale", min(cand), if (min(cand) < 1e-3) "PASS" else "FAIL", sprintf("%s; %s", names(cand)[which.min(cand)], paste(signif(cand, 3), collapse = ", "))) }
  Xs <- as(mats[["nonneg counts 300x200"]], "CsparseMatrix"); prs <- prcomp(as.matrix(Xs))
  r <- try_(suppressWarnings(RcppML::pca(Xs, k = 5, seed = 1, resource = RES))); if (!is_err(r)) { cand <- c(cols = max(abs(r@d - prs$sdev[1:5] * sqrt(299)) / (prs$sdev[1] * sqrt(299))), rows = { p2 <- prcomp(t(as.matrix(Xs))); max(abs(r@d - p2$sdev[1:5] * sqrt(199)) / (p2$sdev[1] * sqrt(199))) }); rec("pca_sparse_implicit_center", "auto", "sparse", min(cand), if (min(cand) < 1e-3) "PASS" else "FAIL", paste(names(cand), signif(cand, 3), collapse = ", ")) } else rec("pca_sparse_implicit_center", "auto", "sparse", NA, "ERR", r)
  ve <- try_(variance_explained(r)); if (!is_err(ve)) rec("variance_explained", "pca sparse", "sum", sum(ve), "INFO", paste(signif(ve, 4), collapse = " "))
  # constraints: which methods honor them
  A <- as.matrix(mats[["nonneg counts 300x200"]]) + 0
  cons <- list("nonneg" = list(nonneg = TRUE), "L1=0.3" = list(L1 = 0.3), "L1=c(0.3,0)" = list(L1 = c(0.3, 0)), "L2=1" = list(L2 = 1), "upper_bound=0.1" = list(upper_bound = 0.1), "L21=0.3" = list(L21 = 0.3), "angular=0.5" = list(angular = 0.5), "graph_V" = list(graph_V = chain(200), graph_lambda = 5), "graph_U" = list(graph_U = chain(300), graph_lambda = 5), "robust" = list(robust = TRUE), "mask zeros" = list(mask = "zeros"))
  for (meth in methods) { b0 <- try_(suppressWarnings(RcppML::svd(A, k = 4, method = meth, seed = 1, resource = RES)))
    for (cn in names(cons)) { r <- try_(suppressWarnings(do.call(RcppML::svd, c(list(A, k = 4, method = meth, seed = 1, resource = RES), cons[[cn]])))); path <- sprintf("%s | %s", meth, cn)
      if (is_err(r)) { rec("svd_constraint", path, "", NA, "REJECTED", r); next }
      same <- !is_err(b0) && length(r@d) == length(b0@d) && max(abs(abs(r@u) - abs(b0@u))) < 1e-6
      extra <- sprintf("min(u)=%.3g min(v)=%.3g zero frac u=%.3f v=%.3f max|u|=%.3g max|v|=%.3g d=%s", min(r@u), min(r@v), mean(r@u == 0), mean(r@v == 0), max(abs(r@u)), max(abs(r@v)), paste(signif(r@d, 4), collapse = " "))
      st <- if (same) "NOOP" else if (cn == "nonneg" && (min(r@u) < -1e-6 || min(r@v) < -1e-6)) "FAIL" else if (grepl("^L1", cn) && mean(r@u == 0) + mean(r@v == 0) == 0) "FAIL" else if (any(!is.finite(r@d))) "FAIL" else "PASS"
      rec("svd_constraint", path, "", NA, st, extra) } }
  # constrained deflation: redundancy of components and monotone L1 sparsity
  for (kk in c(4, 12)) { r <- try_(suppressWarnings(RcppML::svd(A, k = kk, nonneg = TRUE, seed = 1, resource = RES))); if (is_err(r)) { rec("nonneg_svd_redundancy", sprintf("k=%d", kk), "", NA, "ERR", r); next }
    C <- abs(cor(r@v)); rec("nonneg_svd_redundancy", sprintf("k=%d", kk), "max |cor| between V cols", max(C[upper.tri(C)]), if (max(C[upper.tri(C)]) < 0.95) "PASS" else "FAIL", sprintf("numerical rank of V: %d of %d; d sorted: %s", qr(r@v)$rank, kk, !is.unsorted(rev(r@d)))) }
  sp <- sapply(c(0, 0.1, 0.3, 0.6, 0.9), function(l) { r <- try_(suppressWarnings(RcppML::svd(A, k = 3, L1 = l, seed = 1, resource = RES))); if (is_err(r)) NA else mean(r@v == 0) })
  rec("svd_L1_monotone_sparsity", "deflation/auto", "zero frac of V", NA, if (!anyNA(sp) && all(diff(sp) >= -0.01) && sp[5] > sp[1]) "PASS" else "FAIL", paste(c(0, 0.1, 0.3, 0.6, 0.9), signif(sp, 3), sep = ":", collapse = " "))
  # masked SVD: missing entries must not bias singular values
  X <- mats[["exact rank 5 (30x20)"]]; X <- X[rep(1:30, 10), rep(1:20, 10)] ; set.seed(3); M <- as(matrix(runif(length(X)) < 0.2, nrow(X)) * 1, "CsparseMatrix"); Xm <- X; Xm[as.matrix(M) == 1] <- 0; dref <- base::svd(X)$d[1:5]
  r <- try_(suppressWarnings(RcppML::svd(Xm, k = 5, mask = M, seed = 1, resource = RES))); if (is_err(r)) rec("svd_mask_unbiased", "20% masked rank-5", "", NA, "ERR", r) else rec("svd_mask_unbiased", "20% masked rank-5", "mean d/d_true", mean(r@d / dref), if (abs(mean(r@d / dref) - 1) < 0.03) "PASS" else "FAIL", sprintf("ratios %s (0.8 = masked entries treated as zeros)", paste(signif(r@d / dref, 4), collapse = " ")))
  # projection of new data, reconstruction helper
  r <- try_(RcppML::svd(mats[[1]], k = 5, seed = 1, resource = RES)); if (!is_err(r)) { p <- try_(predict(r, mats[[1]])); if (is_err(p)) rec("svd_predict", "dense", "", NA, "ERR", p) else { cand <- c("U*d" = max(abs(abs(p) - abs(r@u %*% diag(r@d)))), "U" = max(abs(abs(p) - abs(r@u)))); rec("svd_predict", "dense", "predict(train) equals", min(cand), if (min(cand) < 1e-2) "PASS" else "FAIL", sprintf("%s (dim %s)", names(cand)[which.min(cand)], paste(dim(p), collapse = "x"))) } }
}

# =========================================================================================
if (SECTION == "F") {  # clustering, networks, streaming, utilities
  ari <- function(a, b) { t <- table(a, b); n <- sum(t); s <- sum(choose(t, 2)); ra <- sum(choose(rowSums(t), 2)); rb <- sum(choose(colSums(t), 2)); e <- ra * rb / choose(n, 2); (s - e) / ((ra + rb) / 2 - e) }
  set.seed(51); m <- 300; cent <- matrix(rexp(m * 4) * (runif(m * 4) < 0.3), m, 4); lab <- rep(1:4, c(60, 50, 40, 30)); A <- cent[, lab] * matrix(rexp(m * 180, 1), m) ; A <- A + 0.05 * matrix(rexp(m * 180), m); As <- as(A, "CsparseMatrix")
  two <- lab <= 2
  for (inp in c("sparse", "dense")) { X <- if (inp == "sparse") As[, lab %in% c(1, 3)] else A[, lab %in% c(1, 3)]; tl <- lab[lab %in% c(1, 3)]
    b <- try_(bipartition(X, seed = 1)); if (is_err(b)) { rec("bipartition", inp, "", NA, "ERR", b); next }
    cl <- rep(0, ncol(X)); cl[b$samples1 + 1] <- 1
    rec("bipartition_two_clusters", inp, "ARI", ari(cl, tl), if (ari(cl, tl) > 0.95) "PASS" else "FAIL", sprintf("names: %s; sizes %d/%d; dist=%s; index base: min(samples1)=%s min(samples2)=%s", paste(names(b), collapse = ","), length(b$samples1), length(b$samples2), format(b$dist), min(b$samples1), min(b$samples2)))
    b2 <- try_(bipartition(X, seed = 1)); if (!is_err(b2)) rec("bipartition_reproducible", inp, "same seed", NA, if (identical(b$samples1, b2$samples1)) "PASS" else "FAIL") }
  for (inp in c("sparse", "dense")) { X <- if (inp == "sparse") As else A
    d <- try_(dclust(X, min_samples = 10, min_dist = 0.3, seed = 1)); if (is_err(d)) { rec("dclust", inp, "", NA, "ERR", d); next }
    cl <- integer(ncol(X)); for (i in seq_along(d)) { s <- d[[i]]$samples; cl[s + 1] <- i }
    rec("dclust_four_clusters", inp, "ARI", ari(cl, lab), if (ari(cl, lab) > 0.9) "PASS" else "FAIL", sprintf("%d clusters, sizes %s, unassigned %d, index base %d", length(d), paste(lengths(lapply(d, `[[`, "samples")), collapse = ","), sum(cl == 0), min(unlist(lapply(d, `[[`, "samples")))))
    d2 <- try_(dclust(X, min_samples = 10, min_dist = 0.3, seed = 1, threads = 4)); if (!is_err(d2)) rec("dclust_reproducible", inp, "threads 0 vs 4", NA, if (identical(lapply(d, `[[`, "samples"), lapply(d2, `[[`, "samples"))) "PASS" else "FAIL") }
  # k = 2 special case against exact ALS
  set.seed(12); W0 <- matrix(runif(m * 2), m, 2); ref <- als_ref(A[, 1:80], W0, 8); tr <- sapply(1:8, function(it) { r <- fit(A[, 1:80], 2, seed = W0, tol = 0, maxit = it); if (is_err(r)) NA else mse_of(r, A[, 1:80]) })
  rec("rank2_vs_exact_ALS", "dense k=2", "max rel dev", max(abs(tr - ref) / ref), if (!anyNA(tr) && max(abs(tr - ref) / ref) < 0.02) "PASS" else "FAIL", sprintf("pkg %s | exact %s", paste(signif(tr, 4), collapse = " "), paste(signif(ref, 4), collapse = " ")))
  # streaming .spz versus in-memory
  S <- sim(400, 300, 6, noise = 0.2, seed = 8, sparse_frac = 0.4); As2 <- as(S$A, "CsparseMatrix"); f <- tempfile(fileext = ".spz"); w <- try_(st_write(As2, f))
  if (is_err(w)) rec("spz_write", "st_write", "", NA, "ERR", w) else { back <- try_(st_read(f)); if (is_err(back)) rec("spz_roundtrip", "st_read", "", NA, "ERR", back) else rec("spz_roundtrip", "st_write/st_read", "max abs diff", max(abs(back - As2)), if (max(abs(back - As2)) < 1e-6 * max(As2)) "PASS" else "FAIL", sprintf("file %d bytes vs %d in memory", file.size(f), as.integer(object.size(As2))))
    for (sv in c("cd", "cholesky")) for (st in list("auto", TRUE)) { r0 <- fit(As2, 6, seed = 1, solver = sv, maxit = 30, tol = 0); r1 <- fit(f, 6, seed = 1, solver = sv, maxit = 30, tol = 0, streaming = st); lab2 <- sprintf("%s streaming=%s", sv, st)
      if (is_err(r1)) { rec("spz_fit_matches_memory", lab2, "", NA, "ERR", r1); next }
      rec("spz_fit_matches_memory", lab2, "rel mse diff", abs(mse_of(r1, S$A) / mse_of(r0, S$A) - 1), if (abs(mse_of(r1, S$A) / mse_of(r0, S$A) - 1) < 1e-3) "PASS" else "FAIL", sprintf("max|dW|=%.3g", max(abs(r1@w - r0@w)))) }
    for (ls in c("gp", "nb")) { Ac <- as(round(S$A * 4), "CsparseMatrix"); f2 <- tempfile(fileext = ".spz"); st_write(Ac, f2); r0 <- fit(Ac, 6, seed = 1, loss = ls, maxit = 30); r1 <- fit(f2, 6, seed = 1, loss = ls, maxit = 30, streaming = TRUE)
      if (is_err(r1) || is_err(r0)) rec("spz_fit_matches_memory", ls, "", NA, "ERR", if (is_err(r1)) r1 else r0) else rec("spz_fit_matches_memory", ls, "max|dW|", max(abs(r1@w - r0@w)), if (max(abs(r1@w - r0@w)) < 1e-2) "PASS" else "FAIL", "") } }
  # multi-modal shared factorization: list input equals stacked matrix
  A1 <- As2[1:250, ]; A2 <- As2[251:400, ]; r <- try_(suppressWarnings(RcppML::nmf(list(A1, A2), 6, seed = 1, maxit = 50)))
  if (is_err(r)) rec("multimodal_list_input", "nmf(list)", "", NA, "ERR", r) else { cls <- class(r)[1]; h <- try_(if (isS4(r)) r@h else r$h); r0 <- fit(As2, 6, seed = 1, maxit = 50)
    rec("multimodal_list_input", "nmf(list(A1,A2))", "class", NA, "INFO", sprintf("returns %s; names %s", cls, paste(if (isS4(r)) slotNames(r) else names(r), collapse = ",")))
    if (!is_err(h) && is.matrix(h)) { W <- t(nnls(h = h, A = as.matrix(As2))); rec("multimodal_equals_stacked", "list vs rbind", "mse ratio", NA, "INFO", sprintf("stacked-fit mse %.5g", mse_of(r0, S$A))) } }
  for (nm in c("loss", "mask", "L21")) { a <- try_(suppressWarnings(RcppML::nmf(list(A1, A2), 6, seed = 1, maxit = 20))); b <- try_(suppressWarnings(switch(nm, loss = RcppML::nmf(list(round(A1 * 4), round(A2 * 4)), 6, seed = 1, maxit = 20, loss = "gp"), mask = RcppML::nmf(list(A1, A2), 6, seed = 1, maxit = 20, mask = "zeros"), L21 = RcppML::nmf(list(A1, A2), 6, seed = 1, maxit = 20, L21 = c(0.5, 0)))))
    if (is_err(b)) rec("multimodal_option", nm, "", NA, "REJECTED", b) else if (nm != "loss") { ha <- if (isS4(a)) a@h else a$h; hb <- if (isS4(b)) b@h else b$h; if (is.matrix(ha) && is.matrix(hb)) rec("multimodal_option", nm, "max|dH|", max(abs(ha - hb)), if (max(abs(ha - hb)) > 1e-6) "PASS" else "NOOP") } }
  # factor network: deep NMF loss must not increase with more iterations
  dn <- try_({ inp <- factor_input(As2); l1 <- nmf_layer(inp, 12); l2 <- nmf_layer(l1, 4); sapply(c(5, 20, 60), function(it) { net <- factor_net(inp, l2, config = factor_config(maxit = it, tol = 0, seed = 1)); ft <- fit(net); NA }) })
  rec("factor_net_api", "deep nmf", "", NA, if (is_err(dn)) "ERR" else "INFO", if (is_err(dn)) dn else "constructed")
  # symmetric and projective models
  B <- sim(150, 10, 5, noise = 0.02, seed = 3)$A; Sy <- tcrossprod(B); r <- fit(Sy, 5, seed = 1, symmetric = TRUE, maxit = 200, tol = 1e-7)
  if (is_err(r)) rec("symmetric_nmf", "A=BBt", "", NA, "ERR", r) else rec("symmetric_nmf", "A=BBt", "max|W-Ht| ; rel recon err", max(abs(r@w - t(r@h))), if (max(abs(r@w - t(r@h))) < 1e-4) "PASS" else "FAIL", sprintf("rel error %.4g vs plain NMF %.4g", sqrt(sum((Sy - recon(r))^2) / sum(Sy^2)), { p <- fit(Sy, 5, seed = 1, maxit = 200, tol = 1e-7); sqrt(sum((Sy - recon(p))^2) / sum(Sy^2)) }))
  r <- fit(As2, 6, seed = 1, projective = TRUE, maxit = 100); if (is_err(r)) rec("projective_nmf", "sparse", "", NA, "ERR", r) else { Hp <- crossprod(r@w, as.matrix(As2)); cr <- sapply(1:6, function(i) cor(Hp[i, ], r@h[i, ])); p <- fit(As2, 6, seed = 1, maxit = 100)
    rec("projective_nmf", "sparse", "min cor(H, WtA)", min(cr), if (min(cr) > 0.999) "PASS" else "FAIL", sprintf("mse %.5g vs plain %.5g; min(h)=%.3g", mse_of(r, S$A), mse_of(p, S$A), min(r@h))) }
  # utilities against brute-force references
  set.seed(9); C <- matrix(runif(36), 6); bm <- try_(bipartiteMatch(C)); if (is_err(bm)) rec("bipartiteMatch", "6x6", "", NA, "ERR", bm) else { perms <- function(v) if (length(v) == 1) list(v) else do.call(c, lapply(seq_along(v), function(i) lapply(perms(v[-i]), function(p) c(v[i], p)))); bestc <- min(sapply(perms(1:6), function(p) sum(C[cbind(1:6, p)]))); as1 <- bm$assignment + 1
    rec("bipartiteMatch_optimal", "6x6", "cost - brute force", sum(C[cbind(1:6, as1)]) - bestc, if (abs(sum(C[cbind(1:6, as1)]) - bestc) < 1e-9) "PASS" else "FAIL", sprintf("reported cost %s", format(bm$cost))) }
  bmn <- try_(bipartiteMatch(C - 0.5)); rec("bipartiteMatch_negative", "6x6", "", NA, if (is_err(bmn)) "ERR" else "PASS", if (is_err(bmn)) bmn else "")
  x <- runif(20); Y <- matrix(runif(100), 20, 5); refc <- drop(crossprod(x, Y)) / (sqrt(sum(x^2)) * sqrt(colSums(Y^2)))
  for (cs in list(list("vector,matrix", function() cosine(x, Y)), list("matrix,vector", function() cosine(Y, x)), list("matrix,matrix", function() cosine(Y, Y)), list("matrix", function() cosine(Y)), list("sparse,sparse", function() cosine(as(Y, "CsparseMatrix"), as(Y, "CsparseMatrix"))))) { v <- try_(cs[[2]]()); if (is_err(v)) { rec("cosine", cs[[1]], "", NA, "ERR", v); next }
    refv <- if (grepl("vector", cs[[1]])) refc else { Yn <- sweep(Y, 2, sqrt(colSums(Y^2)), "/"); crossprod(Yn) }
    rec("cosine", cs[[1]], "max abs diff", max(abs(as.matrix(v) - as.matrix(refv)[seq_along(as.matrix(v))])), if (length(as.matrix(v)) == length(as.matrix(refv)) && max(abs(as.vector(as.matrix(v)) - as.vector(as.matrix(refv)))) < 1e-6) "PASS" else "FAIL", sprintf("dim %s", paste(dim(as.matrix(v)), collapse = "x"))) }
  # NA handling
  X <- S$A; set.seed(2); na <- sample(length(X), 0.1 * length(X)); Xn <- X; Xn[na] <- NA; X0 <- X; X0[na] <- 0
  rn <- fit(Xn, 6, seed = 1, mask = "NA"); r0 <- fit(X0, 6, seed = 1); Mna <- as(matrix(is.na(Xn) * 1, nrow(X)), "CsparseMatrix"); rm <- fit(X0, 6, seed = 1, mask = Mna)
  if (is_err(rn)) rec("mask_NA", "dense with 10% NA", "", NA, "ERR", rn) else { rec("mask_NA_differs_from_zero_fill", "dense with 10% NA", "max|dW|", max(abs(rn@w - r0@w)), if (max(abs(rn@w - r0@w)) > 1e-6) "PASS" else "NOOP", "zero means NAs were fit as zeros")
    if (!is_err(rm)) rec("mask_NA_equals_explicit_mask", "dense with 10% NA", "max|dW|", max(abs(rn@w - rm@w)), if (max(abs(rn@w - rm@w)) < 1e-4) "PASS" else "FAIL", sprintf("held-out error on the NA cells: mask=NA %.4g, explicit mask %.4g, zero-fill %.4g", mean((recon(rn)[na] - X[na])^2), mean((recon(rm)[na] - X[na])^2), mean((recon(r0)[na] - X[na])^2))) }
  # custom mask must exclude entries: corrupt the masked entries and compare
  Xc <- X; Xc[na] <- 100 * max(X); rc <- fit(Xc, 6, seed = 1, mask = Mna); if (!is_err(rc) && !is_err(rm)) rec("mask_excludes_entries", "dense custom mask", "max|dW| after corrupting masked cells", max(abs(rc@w - rm@w)), if (max(abs(rc@w - rm@w)) < 1e-4) "PASS" else "FAIL", "")
  Xcs <- as(Xc, "CsparseMatrix"); rcs <- fit(Xcs, 6, seed = 1, mask = Mna); rms <- fit(as(X0, "CsparseMatrix"), 6, seed = 1, mask = Mna); if (!is_err(rcs) && !is_err(rms)) rec("mask_excludes_entries", "sparse custom mask", "max|dW| after corrupting masked cells", max(abs(rcs@w - rms@w)), if (max(abs(rcs@w - rms@w)) < 1e-4) "PASS" else "FAIL", "")
  # mask = zeros: fit must depend only on nonzero entries
  Z <- as(S$A * (matrix(runif(length(S$A)), nrow(S$A)) < 0.5), "CsparseMatrix"); rz <- fit(Z, 6, seed = 1, mask = "zeros", maxit = 100, solver = "cd"); if (!is_err(rz)) { nzi <- which(as.matrix(Z) != 0); rz0 <- fit(Z, 6, seed = 1, maxit = 100, solver = "cd")
    rec("mask_zeros_fits_nonzeros", "sparse", "mse on nonzeros: masked / unmasked", mean((recon(rz)[nzi] - as.matrix(Z)[nzi])^2) / mean((recon(rz0)[nzi] - as.matrix(Z)[nzi])^2), if (mean((recon(rz)[nzi] - as.matrix(Z)[nzi])^2) < mean((recon(rz0)[nzi] - as.matrix(Z)[nzi])^2)) "PASS" else "FAIL", sprintf("error on the hidden true values: masked %.4g, unmasked %.4g", mean((recon(rz)[-nzi] - S$A[-nzi])^2), mean((recon(rz0)[-nzi] - S$A[-nzi])^2))) }
  # seeds and inits
  r1 <- fit(As2, 6, seed = 1); r2 <- fit(As2, 6, seed = 2); rec("different_seeds_differ", "sparse", "max|dW|", max(abs(r1@w - r2@w)), if (max(abs(r1@w - r2@w)) > 1e-6) "PASS" else "FAIL")
  for (sd in c("lanczos", "irlba", "randomized", "svd")) { r <- fit(As2, 6, seed = sd); rec("svd_based_init", sd, "mse", if (is_err(r)) NA else mse_of(r, S$A), if (is_err(r)) "ERR" else "INFO", if (is_err(r)) r else sprintf("iter %s (random init: mse %.5g iter %s)", r@misc$iter, mse_of(r1, S$A), r1@misc$iter)) }
  rm_ <- fit(As2, 6, seed = c(1, 2, 3)); rec("multi_seed", "seed=c(1,2,3)", "", NA, if (is_err(rm_)) "ERR" else "INFO", if (is_err(rm_)) rm_ else paste(capture.output(print(rm_@misc$all_inits)), collapse = " | "))
  set.seed(1); H0 <- matrix(runif(6 * 300), 6); ra <- fit(As2, 6, seed = 1, h_init = H0, maxit = 1, tol = 0); rb <- fit(As2, 6, seed = 1, maxit = 1, tol = 0); if (!is_err(ra)) rec("h_init_used", "sparse", "max|dW| vs no h_init", max(abs(ra@w - rb@w)), if (max(abs(ra@w - rb@w)) > 1e-6) "PASS" else "NOOP")
  # tol semantics: iterations versus tol
  it <- sapply(c(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8), function(t) { r <- fit(As2, 6, seed = 1, tol = t, maxit = 500); if (is_err(r)) NA else r@misc$iter }); rec("tol_vs_iterations", "sparse", "iters for tol 1e-2..1e-8", NA, if (!is.unsorted(it)) "PASS" else "FAIL", paste(it, collapse = " "))
  # consensus
  cn <- try_(suppressWarnings(consensus_nmf(As, k = 4, reps = 10, seed = 1))); if (is_err(cn)) rec("consensus_nmf", "4 clusters", "", NA, "ERR", cn) else { cl <- tryCatch(cn$clusters, error = function(e) NULL); ok <- !is.null(cl) && length(cl) == length(lab); rec("consensus_nmf", "4 clusters", "ARI", if (ok) ari(cl, lab) else NA, if (ok && ari(cl, lab) > 0.9) "PASS" else "INFO", sprintf("names %s; length(clusters)=%d vs %d samples, %d features", paste(names(cn), collapse = ","), length(cl), length(lab), nrow(As))) }
  # S4 conveniences
  r <- fit(As2, 6, seed = 1); for (e in list(list("model[1]", function() r[1]), list("model[1:2]", function() r[1:2]), list("model$w", function() r$w), list("model$L1", function() r$L1), list("t(model)", function() t(r)), list("prod(model)", function() prod(r)), list("sparsity(model)", function() sparsity(r)), list("summary(model, group)", function() summary(r, group_by = rep(c("a", "b"), 150))), list("align(model, model)", function() align(r, r)), list("head(model)", function() utils::head(r)))) { v <- try_(e[[2]]()); rec("s4_method", e[[1]], "", NA, if (is_err(v)) "ERR" else "PASS", if (is_err(v)) v else "") }
}

df <- do.call(rbind, rows); write.csv(df, OUT, row.names = FALSE)
cat("\n==== SUMMARY", SECTION, RES, "====\n"); print(table(df$check, df$status))
