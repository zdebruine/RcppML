# Create a classifier guide

Steers a factor toward class-consistent structure using per-class
centroid targets computed from the current factor at each iteration.

## Usage

``` r
guide_classifier(labels, lambda = 1, side = c("H", "W"))
```

## Arguments

- labels:

  Integer vector of class labels (0-indexed). Use -1 for unlabeled
  samples. Length must match the number of columns (H guide) or rows (W
  guide) of the data matrix.

- lambda:

  Guide strength. Positive attracts toward class centroids; negative
  repels. Default 1.0.

- side:

  Which factor to guide: "H" (default, sample embeddings) or "W"
  (feature loadings).

## Value

An `nmf_guide` object to pass to `nmf(guides = ...)`.

## See also

[`guide_external`](https://zdebruine.github.io/RcppML/reference/guide_external.md),
[`guide_callback`](https://zdebruine.github.io/RcppML/reference/guide_callback.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
# Create a classifier guide for 3 classes
labels <- rep(0:2, each = 10)
g <- guide_classifier(labels, lambda = 0.5)
g
#> nmf_guide: classifier (lambda=0.50, side=H)
#>   30 labeled samples, 3 classes
```
