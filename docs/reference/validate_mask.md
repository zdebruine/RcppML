# Validate mask parameter

Validate mask parameter

## Usage

``` r
validate_mask(mask, sparse, has_na)
```

## Arguments

- mask:

  NULL, character (`"zeros"` or `"NA"`), or matrix

- sparse:

  Logical; whether sparse mode is enabled

- has_na:

  Logical; whether data contains NA values

## Value

List with `mask_matrix` (dgCMatrix), `mask_zeros` (logical), and
optionally `mask_na`
