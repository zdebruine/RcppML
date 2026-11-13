library(RcppML); data(hawaiibirds); A <- hawaiibirds; labels <- attr(hawaiibirds, "metadata_h")$island
for (res in c("cpu", "gpu")) {
  b <- nmf(A, k = 8, tol = 1e-4, maxit = 100, seed = 42, resource = res)
  Tm <- compute_target(b@h, labels)
  t <- nmf(A, k = 8, tol = 1e-4, maxit = 100, seed = 42, resource = res, target_H = Tm, target_lambda = 0.10)
  cat(sprintf("%s: target/base loss ratio=%.4f  max|dW|=%.3g\n", res, t@misc$loss / b@misc$loss, max(abs(t@w - b@w))))
}
# does each penalty change the fit on each backend?
n <- nrow(A); G <- (abs(row(diag(n)) - col(diag(n))) == 1) * 1
chk <- function(label, ...) for (res in c("cpu", "gpu")) {
  r <- tryCatch({
    b <- nmf(A, k = 8, seed = 42, resource = res); m <- nmf(A, k = 8, seed = 42, resource = res, ...)
    sprintf("%.3g", max(abs(m@w - b@w)))
  }, error = function(e) paste("ERROR:", conditionMessage(e)))
  cat(sprintf("%-14s %s max|dW|=%s\n", label, res, r))
}
chk("L1", L1 = c(0.1, 0.1)); chk("L2", L2 = c(0.5, 0.5)); chk("L21", L21 = c(0.1, 0))
chk("angular", angular = c(0.5, 0)); chk("graph_W", graph_W = G, graph_lambda = 50)
chk("upper_bound", upper_bound = c(0.5, 0))
