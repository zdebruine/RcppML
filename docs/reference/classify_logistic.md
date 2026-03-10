# Logistic regression classifier for factor embeddings

Trains a multinomial logistic regression on factor embeddings using
[`stats::glm`](https://rdrr.io/r/stats/glm.html) (one-vs-rest for \> 2
classes) and evaluates on test samples with the same comprehensive
metrics as `classify_embedding`.

## Usage

``` r
classify_logistic(
  embedding,
  labels,
  test_fraction = 0.2,
  test_idx = NULL,
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

- seed:

  Random seed for train/test split reproducibility.

## Value

An `fn_classifier_eval` object (same structure as KNN variant).

## See also

[`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md),
[`classify_rf`](https://zdebruine.github.io/RcppML/reference/classify_rf.md)

## Examples

``` r
# \donttest{
# Logistic regression on random embeddings
set.seed(42)
embed <- matrix(rnorm(200), nrow = 40, ncol = 5)
labels <- factor(rep(1:4, each = 10))
eval <- classify_logistic(embed, labels, seed = 1)
eval$accuracy
#> [1] 0.5
# }
```
