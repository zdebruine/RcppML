# Compute a Target Matrix for Guided NMF

Builds a k x n target matrix from class labels, suitable for passing as
`target_H` to
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md). Each
sample column is set to its class centroid (optionally ZCA-whitened), so
that target regularization steers H toward class-discriminative
structure.

## Usage

``` r
compute_target(H, labels, whiten = TRUE)
```

## Arguments

- H:

  k x n embedding matrix (e.g. from an initial NMF fit).

- labels:

  Factor or character/integer vector of class labels (length n). `NA`
  entries are left unguided (zero column).

- whiten:

  Logical; apply OAS-shrinkage ZCA whitening to class centroids before
  broadcasting. Default `TRUE`.

## Value

A numeric k x n matrix. Pass it to
`nmf(..., target_H = T, target_lambda = 0.5)` for enrichment or
`target_lambda = -0.5` for batch removal.

## Details

With positive `target_lambda`, NMF attracts H toward the target (label
enrichment). With negative `target_lambda`, NMF uses PROJ_ADV
eigenvalue-projected adversarial removal to suppress target-correlated
structure (batch removal). See
[`vignette("guided-nmf")`](https://zdebruine.github.io/RcppML/articles/guided-nmf.md)
for mathematical details.

**Algorithm:**

1.  Compute per-class centroids from rows/columns of `H`.

2.  (Optional) Apply Oracle-Approximating Shrinkage (OAS) covariance
    estimation followed by ZCA whitening to the centroids, ensuring
    isotropic class structure.

3.  Broadcast each sample's class centroid back to a k x n matrix.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`refine`](https://zdebruine.github.io/RcppML/reference/refine.md)

## Examples

``` r
data(hawaiibirds)
model <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50)
meta <- attr(hawaiibirds, "metadata_h")

# Build target from class labels
target <- compute_target(model@h, labels = meta$island)

# Enrichment: attract H toward island structure
guided <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50,
              target_H = target, target_lambda = 0.5)

# Batch removal: suppress island-correlated structure
removed <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50,
               target_H = target, target_lambda = -0.5)
```
