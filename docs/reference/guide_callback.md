# Create a callback guide

Steers a factor toward a target computed by a user-supplied R function.
The function is called each ALS iteration with the current factor matrix
and iteration number, and must return a target matrix of matching
dimensions.

## Usage

``` r
guide_callback(fn, lambda = 1, side = c("H", "W"))
```

## Arguments

- fn:

  A function with signature `fn(factor, iter)` that returns a target
  matrix (k x n for H, k x m for W).

- lambda:

  Guide strength. Positive attracts; negative repels. Default 1.0.

- side:

  Which factor to guide: "H" (default) or "W".

## Value

An `nmf_guide` object.

## See also

[`guide_external`](https://zdebruine.github.io/RcppML/reference/guide_external.md),
[`guide_embedding`](https://zdebruine.github.io/RcppML/reference/guide_embedding.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# Guide that steers factors toward a decaying target
g <- guide_callback(
  fn = function(factor, iter) factor * exp(-0.01 * iter),
  lambda = 1.0
)
```
