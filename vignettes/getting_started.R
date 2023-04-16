## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- eval = FALSE------------------------------------------------------------
#  install.packages('RcppML')                     # install CRAN version
#  # devtools::install_github("zdebruine/RcppML") # compile dev version

## ---- message = FALSE, warning = FALSE, results = "hide"----------------------
library(RcppML)
library(Matrix)
library(ggplot2)
library(cowplot)

data(iris)
model <- nmf(iris[,1:4], k = 3, seed = 1)

## -----------------------------------------------------------------------------
model

## ---- warning = FALSE---------------------------------------------------------
methods(class = "nmf")

## -----------------------------------------------------------------------------
species_stats <- summary(model, group_by = iris$Species)
species_stats

## ---- fig.height = 2.5, fig.width = 3-----------------------------------------
plot(species_stats, stat = "sum")

## ---- fig.height = 3, fig.width = 4-------------------------------------------
biplot(model, factors = c(1, 2), group_by = iris$Species)

## ---- results = "hide"--------------------------------------------------------
model2 <- nmf(iris[,1:4], k = 3, seed = 1:10, tol = 1e-5)

## -----------------------------------------------------------------------------
# MSE of model from single random initialization
evaluate(model, iris[,1:4])

# MSE of best model among 10 random restarts
evaluate(model2, iris[,1:4])

## ---- results = "hide"--------------------------------------------------------
data(movielens)
ratings <- movielens$ratings
model_L1 <- nmf(ratings, k = 7, L1 = 0.1, seed = 123, mask_zeros = TRUE)

## -----------------------------------------------------------------------------
sparsity(model_L1)

## ---- results = "hide"--------------------------------------------------------
model_no_L1 <- nmf(ratings, k = 7, L1 = 0, seed = 123, mask_zeros = TRUE)
model_L1_h <-  nmf(ratings, k = 7, L1 = c(0, 0.1), seed = 123, mask_zeros = TRUE)
model_L1_w <-  nmf(ratings, k = 7, L1 = c(0.1, 0), seed = 123, mask_zeros = TRUE)

# summarize sparsity of all models in a data.frame
df <- rbind(sparsity(model_no_L1), sparsity(model_L1_h), sparsity(model_L1_w), sparsity(model_L1))
df$side <- c(rep("none", 14), rep("h only", 14), rep("w only", 14), rep("both", 14))
df$side <- factor(df$side, levels = unique(df$side))

## ---- fig.height = 3, fig.width = 4-------------------------------------------
ggplot(df, aes(x = side, y = sparsity, color = model)) + 
  geom_boxplot(outlier.shape = NA, width = 0.6) + 
  geom_point(position = position_jitterdodge()) + theme_classic() + 
  labs(x = "Regularized side of model", y = "sparsity of model factors")

## -----------------------------------------------------------------------------
# L1 = 0
evaluate(model_no_L1, movielens$ratings, mask = "zeros")

# L1 = 0.1
evaluate(model_L1, movielens$ratings, mask = "zeros")

## ---- results = "hide"--------------------------------------------------------
model_low_L1 <- nmf(movielens$ratings, k = 5, L1 = 0.01, seed = 123)

## -----------------------------------------------------------------------------
# cost of bipartite matching: L1 = 0 vs. L1 = 0.01
bipartiteMatch(1 - cosine(model_no_L1$w, model_low_L1$w))$cost / 10

# cost of bipartite matching: L1 = 0 vs. L1 = 0.1
bipartiteMatch(1 - cosine(model_no_L1$w, model_L1$w))$cost / 10

## ---- results = "hide", fig.width = 6, fig.height = 3-------------------------
data(hawaiibirds)
A <- hawaiibirds$counts

test_grids <- sample(1:ncol(A), ncol(A) / 5)
test_species <- sample(1:nrow(A), nrow(A) * 0.5)

# construct a sparse masking matrix for these species and grids
mask <- matrix(0, nrow(A), ncol(A))
mask[test_species, test_grids] <- 1
mask <- as(mask, "dgCMatrix")

model <- nmf(A, k = 15, mask = mask, tol = 1e-6, seed = 123)

df <- rbind(
  data.frame(
    "observed" = as.vector(A[test_species, test_grids]),
    "predicted" = as.vector(prod(model)[test_species, test_grids]),
    "set" = "test"
  ), data.frame(
    "observed" = as.vector(A[-test_species, -test_grids]),
    "predicted" = as.vector(prod(model)[-test_species, -test_grids]),
    "set" = "train"
  )
)

ggplot(df, aes(observed, predicted, color = set)) + 
  theme_classic() + 
  theme(aspect.ratio = 1) + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1), trans = "sqrt") + 
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1), trans = "sqrt") + 
  geom_point(size = 0.5) + 
  facet_grid(cols = vars(set)) + 
  theme(legend.position = "none")

## -----------------------------------------------------------------------------
data(aml)
cv_data <- crossValidate(aml$data, k = 1:10, reps = 3)
head(cv_data)

## -----------------------------------------------------------------------------
plot(cv_data)

