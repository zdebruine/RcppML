# Random forest classifier for factor embeddings

Trains a random forest on factor embeddings using the randomForest
package (must be installed) and evaluates on test samples.

## Usage

``` r
classify_rf(
  embedding,
  labels,
  test_fraction = 0.2,
  test_idx = NULL,
  ntree = 500L,
  seed = NULL
)
```

## Arguments

- embedding:

  Numeric matrix where rows are samples and columns are features (e.g.,
  `t(result$H)` for sample embeddings).

- labels:

  Integer or factor vector of class labels. Length must equal
  `nrow(embedding)`.

- test_fraction:

  Fraction of samples held out for testing (default 0.2).

- test_idx:

  Optional integer vector of test indices. If provided, `test_fraction`
  is ignored.

- ntree:

  Number of trees (default 500).

- seed:

  Random seed for train/test split reproducibility.

## Value

An `fn_classifier_eval` object (same structure as KNN variant).

## See also

[`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md),
[`classify_logistic`](https://zdebruine.github.io/RcppML/reference/classify_logistic.md)

## Examples

``` r
# \donttest{
# Random forest on random embeddings (requires randomForest package)
if (requireNamespace("randomForest", quietly = TRUE)) {
  set.seed(42)
  embed <- matrix(rnorm(200), nrow = 40, ncol = 5)
  labels <- factor(rep(1:4, each = 10))
  eval <- classify_rf(embed, labels, ntree = 100, seed = 1)
  eval$accuracy
}
#> [1] 0.25
# }
```
