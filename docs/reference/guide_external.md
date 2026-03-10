# Create an external target guide

Steers a factor toward a fixed target matrix. Useful for transfer
learning, prior knowledge, or cross-layer coupling.

## Usage

``` r
guide_external(target, lambda = 1, side = c("H", "W"))
```

## Arguments

- target:

  Target matrix with dimensions matching the guided factor (k × n for H,
  k × m for W).

- lambda:

  Guide strength. Positive attracts; negative repels. Default 1.0.

- side:

  Which factor to guide: "H" (default) or "W".

## Value

An `nmf_guide` object to pass to `nmf(guides = ...)`.

## See also

[`guide_classifier`](https://zdebruine.github.io/RcppML/reference/guide_classifier.md),
[`guide_callback`](https://zdebruine.github.io/RcppML/reference/guide_callback.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# Create an external target guide
target <- matrix(runif(30), nrow = 5, ncol = 6)
g <- guide_external(target, lambda = 2.0)
g
#> nmf_guide: external (lambda=2.00, side=H)
#>   target: 5 x 6
```
