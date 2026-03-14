# Assess Embedding Quality

Unified evaluation of an NMF, SVD, or arbitrary embedding matrix against
one or more label vectors. Computes clustering metrics (ARI, NMI,
Silhouette), classification metrics (KNN/LR/RF accuracy, F1, AUC via
cross-validation), and batch-mixing metrics (donor/batch Silhouette, kNN
entropy).

## Usage

``` r
assess(
  x,
  labels,
  batch = NULL,
  metrics = "all",
  n_folds = 5L,
  classifiers = c("knn", "lr", "rf"),
  k_nn = 15L,
  seed = 42L,
  min_class_size = 10L,
  sil_samples_per_class = 200L,
  batch_knn_k = 50L
)
```

## Arguments

- x:

  An object of class `nmf`, `svd`, or a numeric matrix (samples x
  features). For `nmf` objects, uses `t(diag(d) %*% h)`; for `svd`
  objects, uses `u %*% diag(d)`.

- labels:

  A factor, character, or integer vector of primary labels (e.g. cell
  type). Required for clustering and classification metrics.

- batch:

  Optional factor/character/integer vector of batch labels (e.g.
  donor_id, tissue). When provided, batch-mixing metrics are computed.

- metrics:

  Character vector specifying which metrics to compute. Options:
  `"ari"`, `"nmi"`, `"silhouette"`, `"classification"`,
  `"batch_mixing"`, or `"all"` (default).

- n_folds:

  Number of cross-validation folds for classification (default 5).

- classifiers:

  Character vector of classifiers: `"knn"`, `"lr"`, `"rf"`, or any
  combination (default all three).

- k_nn:

  Number of neighbors for kNN classifier (default 15).

- seed:

  Random seed for reproducibility (default 42).

- min_class_size:

  Minimum number of samples per class. Classes smaller than this are
  dropped before evaluation (default 10).

- sil_samples_per_class:

  Number of reference samples per class for approximate silhouette
  (default 200). Higher = more accurate, slower.

- batch_knn_k:

  Number of neighbors for batch entropy (default 50).

## Value

An S3 object of class `nmf_assessment` containing:

- metrics:

  Named list of all computed metric values

- classification:

  Data frame of per-classifier, per-fold results

- params:

  List recording all parameters used

## Details

Uses GPU-accelerated kernels when available, otherwise falls back to
CPU-based approximate algorithms that avoid O(n^2) distance matrices:

- Silhouette: sampled approximation (O(n \* samples_per_class \* C))

- kNN metrics: brute-force on GPU or sampled on CPU (O(n \* k \* n_ref))

- ARI/NMI: contingency table (O(n), always fast)

## Examples

``` r
# \donttest{
# Assess an NMF model
library(Matrix)
A <- abs(rsparsematrix(200, 100, 0.2))
model <- nmf(A, 5, seed = 1)
labels <- factor(sample(letters[1:4], 100, replace = TRUE))
result <- assess(model, labels)
print(result)
#> Embedding Assessment
#>   100 samples, 5 features, 4 classes
#> 
#> Clustering:
#>   NMI:         0.0218
#>   ARI:         -0.0099
#>   Silhouette:  -0.0254
#> 
#> Classification (stratified 5-fold CV):
#>   Mean Accuracy: 0.2828
#>   Mean F1:       0.2409
#>   Mean AUROC:    0.4794
#>     knn   Acc=0.2883  F1=0.2416
#>     lr    Acc=0.2893  F1=0.2394
#>     rf    Acc=0.2707  F1=0.2418

# Select specific metrics
result <- assess(model, labels, metrics = c("nmi", "ari"))

# Include batch assessment
batch <- factor(sample(c("A", "B"), 100, replace = TRUE))
result <- assess(model, labels, batch = batch, metrics = "all")
# }
```
