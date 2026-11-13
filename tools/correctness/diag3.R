# Where does one CPU iteration depart from exact ALS on the nearly exact low-rank matrix?
suppressPackageStartupMessages({ library(Matrix); library(RcppML) })
nnls_ref <- function(G, b, it = 20000, tol = 1e-15) { k <- length(b); x <- rep(0, k)
  for (s in seq_len(it)) { mx <- 0; for (i in seq_len(k)) { xi <- max(0, x[i] + (b[i] - sum(G[i, ] * x)) / G[i, i]); mx <- max(mx, abs(xi - x[i])); x[i] <- xi }; if (mx < tol * (1 + max(abs(x)))) break }; x }
solveH <- function(W, A) { G <- crossprod(W); B <- crossprod(W, A); apply(B, 2, function(b) nnls_ref(G, b)) }
mk <- function(m, n, k, seed) { set.seed(seed); W <- matrix(abs(rnorm(m * k)), m, k); H <- matrix(abs(rnorm(k * n)), k, n)
  noise <- rsparsematrix(m, n, nnz = as.integer(m * n * 0.05)); noise@x <- abs(noise@x) * 0.01; as.matrix(W %*% H + noise) }
chk <- function(label, A, k, sv) { m <- nrow(A); n <- ncol(A); set.seed(5); W0 <- matrix(runif(m * k), m, k)
  H1 <- solveH(W0, A); W1 <- t(solveH(t(H1), t(A))); ref <- mean((A - W1 %*% H1)^2); H2 <- solveH(W1, A); ref15 <- mean((A - W1 %*% H2)^2)
  r <- nmf(A, k, seed = W0, maxit = 1, tol = 0, solver = sv, resource = "cpu"); Wp <- r@w %*% diag(r@d, k); Hp <- r@h
  loss <- mean((A - Wp %*% Hp)^2)
  # is the returned H optimal for the returned W, and the returned W optimal for the returned H?
  Hopt <- solveH(Wp, A); gH <- mean((A - Wp %*% Hp)^2) / mean((A - Wp %*% Hopt)^2) - 1
  Wopt <- t(solveH(t(Hp), t(A))); gW <- mean((A - Wp %*% Hp)^2) / mean((A - Wopt %*% Hp)^2) - 1
  # does the package's H span match the reference's first H (up to row scale)?
  cH <- min(sapply(1:k, function(i) max(abs(cor(Hp[i, ], t(H1))))))
  cat(sprintf("%-30s %-8s loss/ref(1 it)=%.4f loss/ref(1.5 it)=%.4f  gap H|W=%.2e gap W|H=%.2e  min cor(H, H1ref)=%.6f  nnz frac H ref=%.3f pkg=%.3f\n",
              label, sv, loss / ref, loss / ref15, gH, gW, cH, mean(H1 > 0), mean(Hp > 0)))
}
for (sv in c("cd", "cholesky")) {
  chk("300x1000 k=4 near-exact", mk(300, 1000, 4, 77), 4, sv)
  chk("1000x300 k=4 near-exact", t(mk(300, 1000, 4, 77)), 4, sv)
  chk("300x1000 k=4 noisy", pmax(mk(300, 1000, 4, 77) + matrix(rnorm(3e5, sd = 0.5), 300), 0), 4, sv)
  chk("120x80 k=6 near-exact", mk(120, 80, 6, 3), 6, sv)
}
