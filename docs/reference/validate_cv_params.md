# Validate cross-validation parameters

Validate cross-validation parameters

## Usage

``` r
validate_cv_params(test_fraction, patience)
```

## Arguments

- test_fraction:

  Numeric in `[0, 1)` for CV test fraction

- patience:

  Numeric \> 0 for early stopping patience

## Value

List with validated `test_fraction` and `patience`
