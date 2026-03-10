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
if (FALSE) { # \dontrun{
# After fitting a guided NMF:
res <- fit(net)
H <- t(res$L1$H)  # samples x factors
eval <- classify_embedding(H, labels, test_fraction = 0.2, k = 5)
print(eval)
} # }
```
