library(RcppML); data(aml); A <- as.matrix(aml)
chain <- function(n) (abs(row(diag(n)) - col(diag(n))) == 1) * 1
GW <- chain(nrow(A)); GH <- chain(ncol(A))
f <- function(loss, ...) RcppML::nmf(A, k = 4L, loss = loss, seed = 1L, verbose = FALSE, ...)
d <- function(a, b, slot = "w") sprintf("%.3g", max(abs(slot(a, slot) - slot(b, slot))))
for (loss in c("mse", "gp", "nb", "gamma", "inverse_gaussian", "tweedie")) {
  p <- tryCatch(f(loss), error = function(e) e)
  if (inherits(p, "error")) { cat(loss, "plain fit ERROR:", conditionMessage(p), "\n"); next }
  one <- function(label, slot = "w", ...) {
    r <- tryCatch(d(p, f(loss, ...), slot), error = function(e) paste("ERROR:", conditionMessage(e)))
    paste0(label, "=", r)
  }
  cat(sprintf("%-17s max|change| vs plain: %s\n", loss, paste(
    one("graph_W", graph_W = GW, graph_lambda = 50),
    one("graph_H", "h", graph_H = GH, graph_lambda = 50),
    one("angular", angular = c(0.5, 0)),
    one("L1", L1 = c(0.1, 0)),
    one("L21", L21 = c(0.1, 0)),
    one("L2", L2 = c(0.5, 0)),
    one("upper_bound", upper_bound = c(0.5, 0)))))
}
