# Create a reference guide

Steers a factor toward another layer's output matrix. The target changes
each ALS iteration as the referenced layer updates. Used for cross-layer
consistency or hierarchical alignment.

## Usage

``` r
guide_reference(layer_name, lambda = 1, side = c("H", "W"))
```

## Arguments

- layer_name:

  Name of the referenced layer in the factor_net.

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
# Reference guide coupling to another layer
g <- guide_reference("L1", lambda = 0.5, side = "H")
g
#> nmf_guide: reference (lambda=0.50, side=H)
#>   reference to layer 'L1'
```
