## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## -----------------------------------------------------------------------------
library(RcppML)
A <- r_sparsematrix(1000, 1000, inv_density = 16)
not_reproducible_model <- nmf(A, k = 10, seed = NULL) # default
reproducible_model <- nmf(A, k = 10, seed = 123)

## ---- message = FALSE, warning = FALSE----------------------------------------
data(hawaiibirds)
m1 <- nmf(hawaiibirds$counts, k = 10, seed = 1)
m2 <- nmf(hawaiibirds$counts, k = 10, seed = 1:10)

## -----------------------------------------------------------------------------
evaluate(m1, hawaiibirds$counts)

## -----------------------------------------------------------------------------
evaluate(m2, hawaiibirds$counts)

## -----------------------------------------------------------------------------
A <- hawaiibirds$counts
grids1 <- sample(1:ncol(A), floor(ncol(A)/2))
grids2 <- (1:ncol(A))[-grids1]

model1 <- nmf(A[, grids1], k = 15, seed = 123)
model2 <- nmf(A[, grids2], k = 15, seed = model1@w)

cat("model 1 iterations: ", model1@misc$iter,
    ", model 2 iterations: ", model2@misc$iter)

## -----------------------------------------------------------------------------
model2_new <- nmf(A[, grids2], k = 15, seed = 123)

cat("model 2 warm-start", evaluate(model2_new, A[, grids2]),
    ", model 2 random start: ", evaluate(model2, A[, grids2]))

