# Create a training logger for factor network fitting

Returns a logger object that records per-iteration metrics during
[`fit()`](https://zdebruine.github.io/RcppML/reference/fit.md). After
training, the log can be printed, plotted, or exported to CSV.

## Usage

``` r
training_logger(
  log_loss = TRUE,
  log_norms = FALSE,
  log_classifier = NULL,
  interval = 1L
)
```

## Arguments

- log_loss:

  Log reconstruction loss per layer (default TRUE).

- log_norms:

  Log factor W/H norms per layer (default FALSE).

- log_classifier:

  Evaluate classifier accuracy per iteration using a supplied
  `classify_embedding` configuration. Must be a list with `labels` and
  optionally `test_idx`, `k`, `side`, and `layer` (default NULL =
  disabled).

- interval:

  Log every `interval` iterations (default 1).

## Value

A `training_logger` object to pass to
[`fit()`](https://zdebruine.github.io/RcppML/reference/fit.md).

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`fit`](https://zdebruine.github.io/RcppML/reference/fit.md)

## Examples

``` r
if (FALSE) { # \dontrun{
logger <- training_logger(
  log_norms = TRUE,
  log_classifier = list(labels = labels, test_idx = test_idx, k = 5)
)
res <- fit(net, logger = logger)
print(logger)
plot(logger)
export_log(logger, "training_log.csv")
} # }
```
