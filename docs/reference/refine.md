# Refine an NMF Model Using Label-Guided Correction

Post-hoc correction of an NMF embedding to improve class separation or
remove batch effects, optionally followed by W-refit cycles that
propagate the corrected H back through the model.

## Usage

``` r
refine(
  x,
  data = NULL,
  labels,
  batch = NULL,
  lambda = 0.8,
  cycles = 0L,
  nonneg = TRUE,
  whiten = TRUE
)
```

## Arguments

- x:

  An NMF model (S4 object of class `nmf`) or a k x n embedding matrix.

- data:

  Original data matrix (required when `cycles > 0` or `batch` is
  supplied with `cycles > 0`). Must be the same matrix used to fit `x`.

- labels:

  Factor or character/integer vector of class labels (length n).

- batch:

  Optional factor of batch labels (length n) for batch removal. When
  supplied, batch-correlated structure is suppressed using the PROJ_ADV
  method (eigenvalue-projected adversarial removal). See
  [`vignette("guided-nmf")`](https://zdebruine.github.io/RcppML/articles/guided-nmf.md)
  for details.

- lambda:

  Correction strength in `[0, 1]`. Default 0.8. Controls both label
  enrichment (positive direction) and batch removal strength (when
  `batch` is supplied).

- cycles:

  Number of W-refit cycles. 0 = post-hoc only (default). Each cycle:
  refit W from corrected H, refit H from new W, re-correct H.

- nonneg:

  Enforce non-negativity on refitted factors. Default `TRUE`.

- whiten:

  Apply OAS-ZCA whitening to class centroids. Default `TRUE`.

## Value

If `x` is an `nmf` object, returns an updated `nmf` object. If `x` is a
matrix, returns a corrected k x n matrix.

## Details

The correction proceeds in two stages:

**Stage 1: Post-hoc centroid correction** (always performed)

Computes a target matrix from `labels` via
[`compute_target`](https://zdebruine.github.io/RcppML/reference/compute_target.md),
then shifts each sample's embedding toward its class centroid:
\$\$H\_{corrected} = H + \lambda \cdot T\$\$ where T is the target
matrix (class centroid shifts, optionally whitened).

**Stage 1b: Batch removal** (when `batch` is supplied)

Uses PROJ_ADV (Projected Adversarial) method: computes a batch target
from batch labels with negative `target_lambda`, which subtracts the
batch Gram matrix from the NNLS Gram matrix, eigendecomposes, and clips
negative eigenvalues. This suppresses batch-correlated directions while
preserving all other structure.

**Stage 2: W-refit cycles** (when `cycles > 0`)

Iteratively refits the model:

1.  Solve for W given corrected H: \\W = \arg\min \\A - W d H_c\\\_F^2\\

2.  Solve for H given new W: \\H = \arg\min \\A - W\_{new} d H\\\_F^2\\

3.  Re-apply centroid correction to the new H

## See also

[`compute_target`](https://zdebruine.github.io/RcppML/reference/compute_target.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
data(hawaiibirds)
model <- nmf(hawaiibirds, k = 4, seed = 42, maxit = 50)
meta <- attr(hawaiibirds, "metadata_h")

# Post-hoc correction only (cycles = 0)
corrected <- refine(model, labels = meta$island, lambda = 0.8)

# W-refit cycles: propagate correction back through the model
refined <- refine(model, data = hawaiibirds, labels = meta$island,
                  lambda = 0.8, cycles = 3)
```
