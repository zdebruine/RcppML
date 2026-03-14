# Print a classifier evaluation result

Displays a human-readable summary of classification metrics including
accuracy, macro/weighted F1, AUC, and per-class precision/recall.

## Usage

``` r
# S3 method for class 'fn_classifier_eval'
print(x, ...)
```

## Arguments

- x:

  An `fn_classifier_eval` object returned by
  [`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md),
  [`classify_logistic`](https://zdebruine.github.io/RcppML/reference/classify_logistic.md),
  or
  [`classify_rf`](https://zdebruine.github.io/RcppML/reference/classify_rf.md).

- ...:

  Additional arguments (unused).

## Value

Invisibly returns `x`.

## See also

[`summary.fn_classifier_eval`](https://zdebruine.github.io/RcppML/reference/summary.fn_classifier_eval.md),
[`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md)
