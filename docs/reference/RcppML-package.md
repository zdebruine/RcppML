# RcppML: Fast Non-Negative Matrix Factorization and Divisive Clustering

High-performance non-negative matrix factorization (NMF), singular value
decomposition (SVD/PCA), and divisive clustering for large sparse and
dense matrices, powered by Rcpp and Eigen.

## NMF (Non-negative Matrix Factorization)

- [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md):

  Fit NMF model (sparse or dense input, optional cross-validation)

- [`evaluate`](https://zdebruine.github.io/RcppML/reference/evaluate.md):

  Evaluate reconstruction loss of an NMF model

- [`align`](https://zdebruine.github.io/RcppML/reference/align.md):

  Align factors across NMF models

- [`predict,nmf-method`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md):

  Project new data onto a fitted NMF model

- [`consensus_nmf`](https://zdebruine.github.io/RcppML/reference/consensus_nmf.md):

  Consensus clustering from multiple NMF runs

- [`simulateNMF`](https://zdebruine.github.io/RcppML/reference/simulateNMF.md):

  Simulate data from a known NMF model

- [`auto_nmf_distribution`](https://zdebruine.github.io/RcppML/reference/auto_nmf_distribution.md):

  Select distribution based on data characteristics

## SVD / PCA

- [`svd`](https://zdebruine.github.io/RcppML/reference/svd.md):

  Truncated SVD via deflation

- [`pca`](https://zdebruine.github.io/RcppML/reference/pca.md):

  PCA (centered SVD)

- [`reconstruct`](https://zdebruine.github.io/RcppML/reference/svd-class.md):

  Reconstruct matrix from SVD/PCA model

- [`variance_explained`](https://zdebruine.github.io/RcppML/reference/svd-class.md):

  Proportion of variance per factor

## NNLS (Non-negative Least Squares)

- [`nnls`](https://zdebruine.github.io/RcppML/reference/nnls.md):

  Solve non-negative least squares problems

## Clustering

- [`dclust`](https://zdebruine.github.io/RcppML/reference/dclust.md):

  Divisive clustering via recursive rank-2 NMF

- [`bipartition`](https://zdebruine.github.io/RcppML/reference/bipartition.md):

  Split samples into two groups via rank-2 NMF

- [`bipartiteMatch`](https://zdebruine.github.io/RcppML/reference/bipartiteMatch.md):

  Match two sets of cluster labels

## Factor Networks (multi-layer / multi-modal)

- [`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md):

  Compile a factorization network

- [`fit`](https://zdebruine.github.io/RcppML/reference/fit.md):

  Fit a compiled factor network

- [`factor_input`](https://zdebruine.github.io/RcppML/reference/factor_input.md),
  [`nmf_layer`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md),
  [`svd_layer`](https://zdebruine.github.io/RcppML/reference/svd_layer.md):

  Node constructors

- [`factor_shared`](https://zdebruine.github.io/RcppML/reference/factor_shared.md),
  [`factor_concat`](https://zdebruine.github.io/RcppML/reference/factor_concat.md),
  [`factor_add`](https://zdebruine.github.io/RcppML/reference/factor_add.md):

  Merge operations

- [`factor_config`](https://zdebruine.github.io/RcppML/reference/factor_config.md),
  [`W`](https://zdebruine.github.io/RcppML/reference/W.md),
  [`H`](https://zdebruine.github.io/RcppML/reference/W.md):

  Configuration

- [`cross_validate_graph`](https://zdebruine.github.io/RcppML/reference/cross_validate_graph.md):

  Cross-validate a factor network

## StreamPress I/O

- [`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
  [`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md):

  Read/write .spz files

- [`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md):

  Inspect .spz file metadata

- [`st_read_obs`](https://zdebruine.github.io/RcppML/reference/st_read_obs.md),
  [`st_read_var`](https://zdebruine.github.io/RcppML/reference/st_read_var.md):

  Read embedded metadata tables

- [`st_read_gpu`](https://zdebruine.github.io/RcppML/reference/st_read_gpu.md),
  [`st_free_gpu`](https://zdebruine.github.io/RcppML/reference/st_free_gpu.md):

  GPU-direct .spz reading

## GPU

- [`gpu_available`](https://zdebruine.github.io/RcppML/reference/gpu_available.md):

  Check GPU availability

- [`gpu_info`](https://zdebruine.github.io/RcppML/reference/gpu_info.md):

  Get GPU device details

## Utilities

- [`cosine`](https://zdebruine.github.io/RcppML/reference/cosine.md):

  Cosine similarity

- [`sparsity`](https://zdebruine.github.io/RcppML/reference/sparsity.md):

  Matrix sparsity fraction

## See also

Useful links:

- <https://github.com/zdebruine/RcppML>

- Report bugs at <https://github.com/zdebruine/RcppML/issues>

## Author

**Maintainer**: Zachary DeBruine <zacharydebruine@gmail.com>
([ORCID](https://orcid.org/0000-0003-2234-4827))
