# Evaluate classification performance of factor embeddings

Trains a k-nearest-neighbor classifier on factor embeddings and
evaluates on held-out test samples. Returns a comprehensive metrics
object.

## Usage

``` r
classify_embedding(
  embedding,
  labels,
  test_fraction = 0.2,
  test_idx = NULL,
  k = 5L,
  seed = NULL,
  distance = c("euclidean", "cosine")
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

- k:

  Number of nearest neighbors (default 5).

- seed:

  Random seed for train/test split reproducibility.

- distance:

  Distance metric: "euclidean" (default) or "cosine".

## Value

An `fn_classifier_eval` object with fields:

- accuracy:

  Overall test accuracy

- per_class:

  Data frame with per-class precision, recall, F1, support

- macro_precision:

  Macro-averaged precision

- macro_recall:

  Macro-averaged recall

- macro_f1:

  Macro-averaged F1

- weighted_f1:

  Support-weighted F1

- auc:

  Macro-averaged one-vs-rest AUC (from neighbor vote fractions)

- confusion:

  Confusion matrix (rows = true, cols = predicted)

- predictions:

  Test set predictions

- test_labels:

  True test labels

- train_idx:

  Training sample indices

- test_idx:

  Test sample indices

- k:

  Number of neighbors used

## See also

[`classify_logistic`](https://zdebruine.github.io/RcppML/reference/classify_logistic.md),
[`classify_rf`](https://zdebruine.github.io/RcppML/reference/classify_rf.md)

## Examples

``` r
# \donttest{
data(digits)
model <- nmf(digits, 10, maxit = 20, seed = 1, verbose = FALSE)
labels <- attr(digits, "target")
eval <- classify_embedding(model$w, labels, test_fraction = 0.2, k = 5, seed = 42)
print(eval)
#> Classification Evaluation (5-NN, euclidean distance)
#>   Samples: 1438 train, 359 test, 10 classes
#>   Accuracy:  0.9499
#>   Macro F1:  0.9491  (P=0.9516, R=0.9499)
#>   Weighted F1: 0.9494
#>   AUC (macro): 0.9949
#> 
#> Per-class:
#>  class precision recall     f1 support
#>      0    0.9355 1.0000 0.9667      29
#>      1    0.8500 0.9444 0.8947      36
#>      2    0.9070 1.0000 0.9512      39
#>      3    0.9429 0.9429 0.9429      35
#>      4    1.0000 0.9500 0.9744      40
#>      5    1.0000 0.9706 0.9851      34
#>      6    1.0000 1.0000 1.0000      39
#>      7    0.9444 1.0000 0.9714      34
#>      8    0.9643 0.7941 0.8710      34
#>      9    0.9722 0.8974 0.9333      39
# }
```
