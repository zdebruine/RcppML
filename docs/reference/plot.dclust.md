# Plot divisive clustering hierarchy

Reconstructs and plots the binary splitting tree from a `dclust` result.
Each cluster's binary path ID encodes its position in the hierarchy
(e.g., `"01"` = root-\>left-\>right). If `labels` are provided, a
stacked composition bar is drawn below each leaf showing label
proportions.

## Usage

``` r
# S3 method for class 'dclust'
plot(
  x,
  labels = NULL,
  palette = NULL,
  main = "Divisive Clustering Hierarchy",
  ...
)
```

## Arguments

- x:

  a `dclust` object (list of clusters with binary path IDs)

- labels:

  optional character or factor vector of class labels, one per sample in
  the original data matrix passed to
  [`dclust`](https://zdebruine.github.io/RcppML/reference/dclust.md)

- palette:

  optional named character vector mapping label levels to colors. If
  `NULL`, generated automatically.

- main:

  plot title

- ...:

  additional arguments passed to
  [`plot.dendrogram`](https://rdrr.io/r/stats/dendrogram.html)

## Value

`x` (invisibly)

## See also

[`dclust`](https://zdebruine.github.io/RcppML/reference/dclust.md)
