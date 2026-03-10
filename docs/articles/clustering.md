# Clustering, Consensus, and Classification

## Why NMF for Clustering?

NMF is inherently a soft clustering method. Each column of H gives a
sample’s “membership” across k factors, and taking the argmax over rows
yields a hard cluster assignment. However, NMF is non-convex — different
random initializations produce different solutions and potentially
different clusters. **Consensus NMF** addresses this by running many
replicates, tracking which samples consistently co-cluster, and building
a robust consensus matrix.

Beyond flat clustering, RcppML provides **divisive hierarchical
clustering** via recursive rank-2 NMF splits, and **classification from
embeddings** that leverages the low-dimensional H representation for
supervised tasks.

## API Reference

### Consensus NMF

``` r
consensus_nmf(data, k, reps = 50, method = c("hard", "knn_jaccard"),
              knn = 10, seed = NULL, threads = 0, verbose = FALSE, ...)
```

- `data` — input matrix (features × samples) or `.spz` path
- `k` — factorization rank (number of clusters)
- `reps` — number of NMF replicates (more = more stable; use ≤ 30 for
  speed)
- `method` — `"hard"` assigns clusters by dominant factor;
  `"knn_jaccard"` uses k-NN Jaccard overlap (more robust to ambiguity)

Returns a `consensus_nmf` object with `$consensus` (n × n co-clustering
matrix), `$clusters` (assignments), `$cophenetic` (stability metric),
and `$models` (all fitted models).

Use [`plot()`](https://rdrr.io/r/graphics/plot.default.html) for a
consensus heatmap and [`summary()`](https://rdrr.io/r/base/summary.html)
for cluster statistics.

### Divisive Clustering

``` r
bipartition(data, tol = 1e-5, nonneg = TRUE, ...)
dclust(A, min_samples, min_dist = 0, tol = 1e-5, maxit = 100,
       nonneg = TRUE, seed = NULL, threads = 0, verbose = FALSE)
```

[`bipartition()`](https://zdebruine.github.io/RcppML/reference/bipartition.md)
performs a single rank-2 NMF split.
[`dclust()`](https://zdebruine.github.io/RcppML/reference/dclust.md)
recursively bipartitions until clusters are smaller than `min_samples`
or have modularity below `min_dist`.

**Important**:
[`dclust()`](https://zdebruine.github.io/RcppML/reference/dclust.md)
returns 0-indexed `$samples`. Add 1 for R-style indexing.

### Factor Matching

``` r
bipartiteMatch(x)
```

Hungarian algorithm for optimal 1:1 factor correspondence given a cost
matrix. Returns 0-indexed `$assignment` and total `$cost`.

### Classification from Embeddings

``` r
classify_embedding(embedding, labels, test_fraction = 0.2, k = 5L, seed = NULL)
classify_logistic(embedding, labels, test_fraction = 0.2, seed = NULL)
```

Both take a samples × features embedding matrix (e.g., `t(model@h)`) and
class labels. Return an `fn_classifier_eval` object with `$accuracy`,
`$macro_f1`, `$confusion`, and per-class metrics.

## Theory

The **consensus matrix** $C$ is defined element-wise: $C_{ij}$ is the
fraction of replicates where samples $i$ and $j$ are assigned to the
same cluster. Perfect clustering produces a binary consensus (0 or 1);
noisy or unstable clustering yields intermediate values.

The **cophenetic correlation** measures how well hierarchical clustering
of the consensus matrix preserves pairwise distances. Higher values
indicate more stable clustering.

**Divisive clustering** recursively applies rank-2 NMF. Each split finds
the best binary partition; the process stops when clusters are too small
or too homogeneous (low modularity).

**Classification from H**: Columns of H are k-dimensional embeddings of
samples. Any classifier (k-NN, logistic regression) applied to these
embeddings can separate classes, assuming they are captured by the NMF
factors.

## Example 1: Cancer Subtype Discovery (Golub Leukemia)

The Golub leukemia dataset contains 38 bone marrow samples — 27 ALL and
11 AML — measured across 5,000 genes. Can unsupervised NMF discover
these known subtypes?

``` r
data(golub)
labels <- attr(golub, "cancer_type")

# Consensus NMF: samples (rows) x genes (columns)
cons <- consensus_nmf(golub, k = 2, reps = 20, seed = 42, verbose = FALSE)
```

``` r
plot(cons, show_clusters = TRUE) +
  ggtitle("Consensus matrix: Golub leukemia (k = 2, 20 replicates)")
```

![](clustering_files/figure-html/golub-heatmap-1.png)

The consensus heatmap reveals two tightly co-clustered groups. Let’s
check how well they correspond to the known ALL/AML labels.

``` r
# Confusion matrix: consensus clusters vs. true labels
conf <- table(Cluster = cons$clusters, Cancer = labels)
knitr::kable(conf, caption = "Consensus clusters vs. true cancer type")
```

| ALL | AML |
|----:|----:|
|  24 |   1 |
|   3 |  10 |

Consensus clusters vs. true cancer type

``` r
summary_df <- data.frame(
  Metric = c("Cophenetic correlation", "Cluster 1 size", "Cluster 2 size"),
  Value = c(round(cons$cophenetic, 4),
            sum(cons$clusters == 1),
            sum(cons$clusters == 2))
)
knitr::kable(summary_df, caption = "Consensus NMF summary statistics")
```

| Metric                 | Value |
|:-----------------------|------:|
| Cophenetic correlation |     1 |
| Cluster 1 size         |    25 |
| Cluster 2 size         |    13 |

Consensus NMF summary statistics

Consensus NMF with k = 2 cleanly separates the two leukemia subtypes.
The cophenetic correlation confirms high clustering stability across
replicates.

## Example 2: Hierarchical Clustering with dclust (AML Chromatin)

The AML dataset contains 824 chromatin accessibility regions across 135
samples with known subtype annotations. Divisive clustering discovers a
hierarchy of subtypes.

``` r
data(aml)
meta <- attr(aml, "metadata_h")

clusters <- dclust(aml, min_samples = 10, min_dist = 0.01, seed = 42)

# Build cluster composition table
cluster_info <- do.call(rbind, lapply(clusters, function(cl) {
  sample_idx <- cl$samples + 1L  # 0-indexed to 1-indexed
  categories <- meta$category[sample_idx]
  top_cats <- sort(table(categories), decreasing = TRUE)
  data.frame(
    Cluster = cl$id,
    Size = cl$size,
    `Top Category` = names(top_cats)[1],
    `Top Count` = as.integer(top_cats[1]),
    `Composition` = paste(paste0(names(top_cats)[1:min(3, length(top_cats))], ":",
                                  as.integer(top_cats[1:min(3, length(top_cats))])),
                          collapse = ", "),
    check.names = FALSE
  )
}))

knitr::kable(head(cluster_info, 8), caption = "Divisive clustering of AML chromatin data")
```

| Cluster | Size | Top Category  | Top Count | Composition                                         |
|--------:|-----:|:--------------|----------:|:----------------------------------------------------|
|       0 |   13 | AML (L-MPP)   |        12 | AML (L-MPP):12, Control (MEP):1                     |
|       1 |   15 | Control (GMP) |         5 | Control (GMP):5, Control (L-MPP):5, Control (MEP):4 |
|       2 |   11 | AML (GMP)     |         7 | AML (GMP):7, AML (MEP):4                            |
|       3 |   21 | AML (GMP)     |        14 | AML (GMP):14, AML (MEP):7                           |
|       4 |   17 | AML (GMP)     |        16 | AML (GMP):16, AML (MEP):1                           |
|       5 |   16 | AML (GMP)     |        14 | AML (GMP):14, AML (MEP):2                           |
|       6 |   16 | AML (GMP)     |        16 | AML (GMP):16                                        |
|       7 |   12 | AML (GMP)     |        12 | AML (GMP):12                                        |

Divisive clustering of AML chromatin data

Divisive clustering discovers a hierarchy of subtypes, with early splits
separating biologically distinct cell populations. Each cluster’s
composition reflects known AML subtype structure.

## Example 3: Comparing Classification Methods

Using the Golub NMF model, we compare classification accuracy from k-NN
and logistic regression applied to the H embedding.

``` r
# Fit NMF to get embeddings
model <- nmf(t(golub), k = 3, seed = 42, maxit = 100)
embedding <- t(model@h)  # 38 samples x 3 factors
int_labels <- as.integer(labels) - 1L  # 0-indexed

# Use same test split for fair comparison
set.seed(42)
n <- nrow(embedding)
test_idx <- sort(sample(n, floor(n * 0.3)))

knn_eval <- classify_embedding(embedding, int_labels, test_idx = test_idx, k = 3, seed = 42)
log_eval <- classify_logistic(embedding, int_labels, test_idx = test_idx, seed = 42)

comp_table <- data.frame(
  Method = c("k-NN (k=3)", "Logistic regression"),
  Accuracy = c(knn_eval$accuracy, log_eval$accuracy),
  `Macro F1` = c(knn_eval$macro_f1, log_eval$macro_f1),
  check.names = FALSE
)
knitr::kable(comp_table, digits = 3,
             caption = "Classification from NMF embeddings (k = 3)")
```

| Method              | Accuracy | Macro F1 |
|:--------------------|---------:|---------:|
| k-NN (k=3)          |        1 |        1 |
| Logistic regression |        1 |        1 |

Classification from NMF embeddings (k = 3)

Both classifiers achieve high accuracy from just 3 NMF factors,
confirming that the low-dimensional embedding captures the ALL/AML
distinction effectively. The NMF representation compresses 5,000 genes
into a handful of components without losing discriminative power.

## What’s Next

- *See the [Factor
  Graphs](https://zdebruine.github.io/RcppML/articles/factor-graphs.md)
  vignette for semi-supervised NMF with `guide_classifier` to improve
  embeddings with labels.*
- *See the
  [Cross-Validation](https://zdebruine.github.io/RcppML/articles/cross-validation.md)
  vignette for choosing the optimal k for clustering.*
- *See the [NMF
  Fundamentals](https://zdebruine.github.io/RcppML/articles/nmf-fundamentals.md)
  vignette for the core API.*
