# Post-Hoc Embedding Correction via OAS-ZCA Whitened Centroid

Corrects an NMF or SVD embedding to enhance class-discriminative
structure and optionally remove batch effects, without re-fitting the
model. Uses Oracle Approximating Shrinkage (OAS) ZCA whitening to
decorrelate the embedding, then shifts each sample toward its class
centroid and away from the grand mean. Optionally projects out
confound-predictive directions (nullspace removal) for batch correction.

This is a \*\*post-hoc\*\* operation: it takes a fitted embedding and
adjusts it for downstream tasks like classification, clustering, or
visualization. Unlike in-NMF guides, this approach never interferes with
NMF convergence and is mathematically guaranteed to improve
centroid-based class separability.

## Usage

``` r
guide_embedding(x, labels, batch = NULL, lambda_c = 0.8, gpu = FALSE)
```

## Arguments

- x:

  An `nmf` object, `svd` object, or \\n \times k\\ numeric matrix of
  sample embeddings (rows = samples, columns = factors).

- labels:

  Factor or integer vector of class labels (length \\n\\). Samples with
  `NA` labels are left uncorrected in the centroid step.

- batch:

  Optional factor or integer vector of batch/confound labels (length
  \\n\\). When provided, confound-predictive directions are projected
  out of the embedding (nullspace removal).

- lambda_c:

  Centroid shift strength in \\\[0, 1\]\\. Controls the interpolation
  between the original sample position (`0`) and the class centroid
  (`1`). Default `0.8`.

- gpu:

  Logical; if `TRUE` and a GPU is available, use the CUDA
  implementation. Default `FALSE` (CPU path).

## Value

If `x` is an `nmf` or `svd` object, returns a modified copy with
corrected embedding. If `x` is a matrix, returns the corrected \\n
\times k\\ matrix.

## Details

The algorithm proceeds in three steps:

1.  **OAS-ZCA Whitening**: Compute the \\k \times k\\ sample covariance
    \\\Sigma\\, estimate the optimal shrinkage intensity \\\alpha^\*\\
    via the Oracle Approximating Shrinkage formula (Chen et al. 2010),
    form the regularized covariance \\\Sigma\_{oas} =
    (1-\alpha^\*)\Sigma + \alpha^\* \frac{\mathrm{tr}(\Sigma)}{k} I\\,
    and apply the ZCA whitening transform \\W = \Sigma\_{oas}^{-1/2}\\.

2.  **Centroid Shift**: In the whitened space, compute per-class
    centroids and the grand mean, then shift each sample: \\h_i
    \leftarrow h_i + \lambda_c (\mu\_{y_i} - \bar{\mu})\\ where
    \\\mu\_{y_i}\\ is the centroid of sample \\i\\'s class and
    \\\bar{\mu}\\ is the grand mean.

3.  **Nullspace Removal** (optional, when `batch` is provided): Compute
    batch centroids, extract the between-batch subspace via SVD, and
    project it out of the embedding: \\H \leftarrow H (I - V V^T)\\
    where \\V\\ spans the batch-predictive directions.

The OAS shrinkage intensity is computed analytically: \$\$\alpha^\* =
\frac{(1 - 2/k)\\\mathrm{tr}(\Sigma^2) + \mathrm{tr}(\Sigma)^2}{(n + 1 -
2/k)(\mathrm{tr}(\Sigma^2) - \mathrm{tr}(\Sigma)^2/k)}\$\$ This is
well-defined for all \\k \geq 2\\ and clipped to \\\[0, 1\]\\.

## Note

Can also be applied automatically by passing `guide_embedding`,
`guide_embedding_batch`, and `guide_embedding_lambda` as advanced
parameters to
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md),
[`pca`](https://zdebruine.github.io/RcppML/reference/pca.md), or
[`factor_config`](https://zdebruine.github.io/RcppML/reference/factor_config.md).

## References

Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage
algorithms for MMSE covariance estimation. *IEEE Transactions on Signal
Processing*, 58(10), 5016–5029.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`svd`](https://zdebruine.github.io/RcppML/reference/svd.md),
[`factor_config`](https://zdebruine.github.io/RcppML/reference/factor_config.md),
[`assess`](https://zdebruine.github.io/RcppML/reference/assess.md),
[`classify_embedding`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md)

## Examples

``` r
# \donttest{
# Fit NMF, then correct embedding for classification
data(hawaiibirds)
model <- nmf(hawaiibirds, k = 8, seed = 42, maxit = 100)
labels <- factor(colnames(hawaiibirds))  # island labels
corrected <- guide_embedding(model, labels, lambda_c = 0.8)

# Works with raw matrices too
H <- t(model@d * model@h)  # n x k embedding
H_corrected <- guide_embedding(H, labels)
# }
```
