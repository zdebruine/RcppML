# Plot Consensus Matrix Heatmap

Plot Consensus Matrix Heatmap

## Usage

``` r
# S3 method for class 'consensus_nmf'
plot(
  x,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  show_clusters = TRUE,
  color_palette = c("white", "#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"),
  interactive = FALSE,
  ...
)
```

## Arguments

- x:

  consensus_nmf object

- cluster_rows:

  whether to reorder rows by hierarchical clustering (default TRUE)

- cluster_cols:

  whether to reorder columns (default TRUE, same as rows)

- show_clusters:

  whether to show cluster assignments as sidebar (default TRUE)

- color_palette:

  color palette name or vector of colors

- interactive:

  whether to make interactive plotly heatmap (default FALSE)

- ...:

  additional arguments (unused)

## Value

A `ggplot2` object (or `plotly` object if `interactive = TRUE`) showing
the consensus heatmap.

## See also

[`consensus_nmf`](https://zdebruine.github.io/RcppML/reference/consensus_nmf.md),
[`summary.consensus_nmf`](https://zdebruine.github.io/RcppML/reference/summary.consensus_nmf.md)

## Examples

``` r
# \donttest{
library(Matrix)
A <- rsparsematrix(50, 30, 0.3)
cons <- consensus_nmf(A, k = 3, reps = 5, seed = 42)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  plot(cons)
}

# }
```
