# Export training log to CSV

Export training log to CSV

## Usage

``` r
export_log(logger, file)
```

## Arguments

- logger:

  A `training_logger` object.

- file:

  Path to write the CSV file.

## Value

Invisibly returns the data frame written.

## See also

[`training_logger`](https://zdebruine.github.io/RcppML/reference/training_logger.md),
[`as.data.frame.training_logger`](https://zdebruine.github.io/RcppML/reference/as.data.frame.training_logger.md)

## Examples

``` r
# \donttest{
logger <- training_logger()
# After fitting: export_log(logger, tempfile(fileext = ".csv"))
# }
```
