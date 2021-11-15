## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- message = FALSE, warning = FALSE----------------------------------------
library(RcppML)
data(hawaiibirds)
m1 <- nmf(hawaiibirds$counts, k = 10, seed = 1)
m2 <- nmf(hawaiibirds$counts, k = 10, seed = 1:10)

## -----------------------------------------------------------------------------
evaluate(m1, hawaiibirds$counts)

## -----------------------------------------------------------------------------
evaluate(m2, hawaiibirds$counts)

## -----------------------------------------------------------------------------
data(movielens)
A <- movielens$ratings
users1 <- sample(1:ncol(A), floor(ncol(A)/2))
users2 <- (1:ncol(A))[-users1]

model1 <- nmf(A[, users1], k = 7, mask = "zeros", seed = 1:10)
model2 <- nmf(A[, users2], k = 7, mask = "zeros", seed = model1@w)

cat("model 1 iterations: ", model1@misc$iter,
    ", model 2 iterations: ", model2@misc$iter)

## -----------------------------------------------------------------------------
model2_new <- nmf(A[, users2], k = 7, mask = "zeros", seed = 1:10)

cat("model 2 warm-start", evaluate(model2_new, A[, users2], mask = "zeros"),
    ", model 2 random start: ", evaluate(model2, A[, users2], mask = "zeros"))

