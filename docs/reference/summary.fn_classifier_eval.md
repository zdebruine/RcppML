# Summarize a classifier evaluation result

Returns a tidy data frame of aggregate classification metrics.

## Usage

``` r
# S3 method for class 'fn_classifier_eval'
summary(object, ...)
```

## Arguments

- object:

  An `fn_classifier_eval` object returned by
  [`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md),
  [`classify_logistic`](https://zdebruine.github.io/RcppML/reference/classify_logistic.md),
  or
  [`classify_rf`](https://zdebruine.github.io/RcppML/reference/classify_rf.md).

- ...:

  Additional arguments (unused).

## Value

A data frame with columns `metric` and `value`.

## See also

[`print.fn_classifier_eval`](https://zdebruine.github.io/RcppML/reference/print.fn_classifier_eval.md),
[`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md)
