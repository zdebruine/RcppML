# Validate mask parameter

Supports the extended mask syntax:

- `NULL`: no masking

- `"zeros"`: mask zero entries (sets mask_zeros=TRUE)

- `"NA"`: mask NA entries

- dgCMatrix or matrix: custom mask matrix

- `list("zeros", <matrix>)`: mask zeros AND custom mask simultaneously

## Usage

``` r
validate_mask(mask, has_na = FALSE)
```

## Arguments

- mask:

  NULL, character, matrix, or list

- has_na:

  Logical; whether data contains NA values

## Value

List with `mask_matrix` (dgCMatrix), `mask_zeros` (logical), and
optionally `mask_na`
