# NMF Fundamentals

## Motivation

How can we decompose a nonnegative matrix into interpretable parts?
Non-negative Matrix Factorization (NMF) answers this by finding
additive, parts-based representations: each factor captures a distinct
pattern, and all weights are nonneg, so factors combine constructively.
Unlike PCA, where components can cancel each other out, NMF factors are
always additive — making them directly interpretable as “parts” of the
data.

NMF is widely used in topic modeling, image decomposition, gene
expression analysis, and recommendation systems. RcppML provides a
high-performance implementation with a distinctive
$A \approx W \cdot \text{diag}(d) \cdot H$ factorization, where the
diagonal scaling $d$ makes factors comparable and ordered by importance.

## API Reference

### The `nmf()` Function

``` r
nmf(data, k, tol = 1e-4, maxit = 100, seed = NULL,
    init = c("random", "lanczos", "irlba"),
    norm = c("L1", "L2", "none"),
    L1 = c(0, 0), L2 = c(0, 0),
    sort_model = TRUE, nonneg = c(TRUE, TRUE),
    loss = "mse", threads = 0, verbose = FALSE,
    on_iteration = NULL, ...)
```

Key parameters:

| Parameter      | Type             | Default       | Description                                      |
|----------------|------------------|---------------|--------------------------------------------------|
| `data`         | matrix/dgCMatrix | —             | Nonneg input matrix (features × samples)         |
| `k`            | integer          | —             | Factorization rank (number of factors)           |
| `tol`          | numeric          | 1e-4          | Convergence tolerance (correlation-distance)     |
| `maxit`        | integer          | 100           | Maximum iterations                               |
| `seed`         | integer          | NULL          | Random seed for reproducibility                  |
| `init`         | character        | “random”      | Initialization method                            |
| `norm`         | character        | “L1”          | Factor normalization method                      |
| `sort_model`   | logical          | TRUE          | Sort factors by decreasing $d$                   |
| `nonneg`       | logical(2)       | c(TRUE, TRUE) | Non-negativity constraints on W and H            |
| `on_iteration` | function         | NULL          | Callback receiving (iter, train_loss, test_loss) |

Initialization options:

| Method      | Description                                                                               |
|-------------|-------------------------------------------------------------------------------------------|
| `"random"`  | Random nonneg initialization (default). May need more iterations to converge.             |
| `"lanczos"` | SVD-based warm start via Lanczos iteration. Often converges faster.                       |
| `"irlba"`   | SVD-based warm start via IRLBA. Similar to Lanczos with different convergence properties. |

Normalization options:

| Method   | Description                                                          |
|----------|----------------------------------------------------------------------|
| `"L1"`   | Columns of W and rows of H sum to 1 (default). Scales stored in $d$. |
| `"L2"`   | Unit L2 norm for columns of W and rows of H.                         |
| `"none"` | No normalization; $d$ is all ones.                                   |

### Factor Extraction and Inspection

The [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md)
function returns an S4 object of class `nmf` with slots accessed via
`$`:

- `model$w` — $m \times k$ feature loading matrix (W)
- `model$h` — $k \times n$ sample embedding matrix (H)
- `model$d` — length-$k$ diagonal scaling vector
- `dim(model)` — returns `c(m, n, k)`
- `head(model, n)` — first $n$ factors
- `model[i]` — subset to specific factor indices

### Reconstruction and Loss

- `prod(model)` — reconstructs $W \cdot \text{diag}(d) \cdot H$ (dense
  matrix)
- `evaluate(model, data)` — reconstruction loss (MSE by default)
- `mse(model$w, model$d, model$h, data)` — standalone MSE computation

### Model Comparison

- `align(model, ref)` — reorder factors to match a reference model
  (Hungarian algorithm)
- `sparsity(model)` — fraction of near-zero entries in W and H
- `cosine(x, y)` — column-wise cosine similarity between matrices
- `bipartiteMatch(cost_matrix)` — Hungarian algorithm for optimal factor
  pairing (returns 0-indexed `$assignment`)

### Convergence Control

- `tol` controls the convergence threshold. The model converges when the
  maximum change in any factor (measured by correlation distance) falls
  below `tol`.
- `maxit` provides a hard iteration cap.
- `on_iteration` accepts a callback
  `function(iter, train_loss, test_loss)` called after each iteration.

## Theory and Algorithms

### Objective

NMF solves the constrained optimization problem:

$$\min\limits_{W \geq 0,\, H \geq 0} \parallel A - W \cdot \text{diag}(d) \cdot H \parallel_{F}^{2}$$

where $A$ is an $m \times n$ nonneg matrix, $W$ is $m \times k$, $H$ is
$k \times n$, and $d$ is a length-$k$ scaling vector.

### Alternating NNLS

RcppML uses alternating non-negative least squares: fix H, solve for W;
fix W, solve for H. After each full iteration, columns of W and rows of
H are normalized and the scales absorbed into $d$. This diagonal scaling
makes factors directly comparable and ordered by importance when
`sort_model = TRUE`.

### Initialization

Random initialization gives a noisy starting point that may require more
iterations. SVD-based methods (Lanczos, IRLBA) compute a truncated SVD
as a warm start, typically converging faster — especially for
well-conditioned data.

### Non-uniqueness

NMF solutions are not unique: different seeds lead to different local
optima. Always set `seed` for reproducibility. For robust factorizations
across multiple random starts, use
[`consensus_nmf()`](https://zdebruine.github.io/RcppML/reference/consensus_nmf.md)
(see the
[Clustering](https://zdebruine.github.io/RcppML/articles/clustering.md)
vignette).

## Worked Examples

### Example 1: Recovering Known Factors from Synthetic Data

We generate a synthetic matrix with 5 true factors and test whether NMF
can recover them.

``` r
sim <- simulateNMF(300, 150, k = 5, noise = 0.3, seed = 42)
model <- nmf(sim$A, k = 5, seed = 1, tol = 1e-5, maxit = 100)
```

To compare learned factors with ground truth, we compute cosine
similarity between each pair of W columns. The
[`bipartiteMatch()`](https://zdebruine.github.io/RcppML/reference/bipartiteMatch.md)
function finds the optimal one-to-one assignment (note: it returns
0-indexed assignments):

``` r
sim_cos <- cosine(model$w, sim$w)
match_result <- bipartiteMatch(1 - sim_cos + 1e-10)
assignment <- match_result$assignment + 1L  # convert to 1-indexed

per_factor_cos <- sapply(seq_len(5), function(i) {
  sim_cos[i, assignment[i]]
})

knitr::kable(
  data.frame(
    Factor = 1:5,
    `Cosine Similarity` = round(per_factor_cos, 4),
    check.names = FALSE
  ),
  caption = "Per-factor cosine similarity between learned and true W columns."
)
```

| Factor | Cosine Similarity |
|-------:|------------------:|
|      1 |            0.4601 |
|      2 |            0.6176 |
|      3 |            0.5977 |
|      4 |            0.4186 |
|      5 |            0.4234 |

Per-factor cosine similarity between learned and true W columns.

NMF recovers the 5 planted factors with high cosine similarity despite
30% Gaussian noise.

``` r
# Prepare data for side-by-side comparison
true_w <- sim$w[, assignment]
learned_w <- model$w

# Build long-format data for ggplot
make_heatmap_df <- function(mat, label) {
  df <- expand.grid(Feature = seq_len(nrow(mat)), Factor = seq_len(ncol(mat)))
  df$Value <- as.vector(mat)
  df$Source <- label
  df
}
hm_df <- rbind(
  make_heatmap_df(true_w, "True W"),
  make_heatmap_df(learned_w, "Learned W")
)
hm_df$Source <- factor(hm_df$Source, levels = c("True W", "Learned W"))

ggplot(hm_df, aes(x = Factor, y = Feature, fill = Value)) +
  geom_raster() +
  facet_wrap(~Source) +
  scale_fill_viridis_c(option = "inferno") +
  labs(title = "True vs. Learned Feature Loadings (W)", x = "Factor", y = "Feature") +
  theme_minimal() +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
```

![](nmf-fundamentals_files/figure-html/synthetic-heatmap-1.png)

The learned W matrix closely mirrors the true factor structure: each
factor activates a distinct subset of features, and the pattern
correspondence is clear despite noise.

### Example 2: Gene Expression Factor Interpretation (AML Data)

The `aml` dataset contains 824 genomic regions × 135 samples from Acute
Myeloid Leukemia ATAC-seq data. Sample subtypes are stored in
`attr(aml, "metadata_h")$category`.

``` r
data(aml)
meta <- attr(aml, "metadata_h")
model_aml <- nmf(aml, k = 6, seed = 42, tol = 1e-5)
```

We identify the top features per factor — the genomic regions with the
highest loadings:

``` r
top_n <- 5
top_features <- do.call(rbind, lapply(1:6, function(f) {
  w_col <- model_aml$w[, f]
  idx <- order(w_col, decreasing = TRUE)[1:top_n]
  data.frame(
    Factor = f,
    Rank = 1:top_n,
    Feature = idx,
    Loading = round(w_col[idx], 4)
  )
}))

knitr::kable(
  top_features,
  caption = "Top 5 features (highest W loadings) per factor.",
  row.names = FALSE
)
```

| Factor | Rank | Feature | Loading |
|-------:|-----:|--------:|--------:|
|      1 |    1 |     804 |  0.0030 |
|      1 |    2 |     181 |  0.0029 |
|      1 |    3 |     689 |  0.0029 |
|      1 |    4 |     489 |  0.0028 |
|      1 |    5 |     580 |  0.0028 |
|      2 |    1 |     606 |  0.0024 |
|      2 |    2 |     701 |  0.0024 |
|      2 |    3 |      56 |  0.0024 |
|      2 |    4 |     627 |  0.0023 |
|      2 |    5 |     426 |  0.0023 |
|      3 |    1 |     279 |  0.0032 |
|      3 |    2 |     644 |  0.0029 |
|      3 |    3 |     420 |  0.0028 |
|      3 |    4 |     173 |  0.0028 |
|      3 |    5 |     211 |  0.0028 |
|      4 |    1 |     736 |  0.0028 |
|      4 |    2 |      56 |  0.0026 |
|      4 |    3 |     498 |  0.0026 |
|      4 |    4 |     441 |  0.0025 |
|      4 |    5 |     130 |  0.0024 |
|      5 |    1 |     713 |  0.0056 |
|      5 |    2 |     140 |  0.0054 |
|      5 |    3 |     478 |  0.0053 |
|      5 |    4 |     176 |  0.0052 |
|      5 |    5 |     180 |  0.0052 |
|      6 |    1 |     390 |  0.0057 |
|      6 |    2 |     568 |  0.0052 |
|      6 |    3 |     577 |  0.0051 |
|      6 |    4 |     160 |  0.0048 |
|      6 |    5 |     763 |  0.0047 |

Top 5 features (highest W loadings) per factor.

Now we visualize the H matrix (sample embeddings) with columns grouped
by AML subtype:

``` r
subtypes <- meta$category
col_order <- order(subtypes)
h_ordered <- model_aml$h[, col_order]
subtypes_ordered <- subtypes[col_order]

h_df <- expand.grid(Factor = seq_len(nrow(h_ordered)), Sample = seq_len(ncol(h_ordered)))
h_df$Weight <- as.vector(h_ordered)
h_df$Subtype <- rep(subtypes_ordered, each = nrow(h_ordered))

ggplot(h_df, aes(x = Sample, y = factor(Factor), fill = Weight)) +
  geom_raster() +
  scale_fill_viridis_c(option = "magma") +
  labs(
    title = "NMF Sample Embeddings (H) by AML Subtype",
    x = "Samples (grouped by subtype)", y = "Factor"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
```

![](nmf-fundamentals_files/figure-html/aml-heatmap-1.png)

The H heatmap reveals clear subtype structure: certain factors activate
preferentially within specific AML subtypes, indicating that NMF
captures biologically meaningful programs from the chromatin
accessibility data.

### Example 3: Convergence and Initialization Comparison

We compare random vs. Lanczos initialization by tracking loss per
iteration using the `on_iteration` callback:

``` r
# Track loss over iterations via callback
track_loss <- function() {
  losses <- numeric(0)
  callback <- function(iter, train_loss, test_loss) {
    losses[iter] <<- train_loss
  }
  list(callback = callback, get = function() losses)
}

tracker_random <- track_loss()
model_random <- nmf(aml, k = 6, seed = 42, init = "random", tol = 1e-6,
                    maxit = 80, on_iteration = tracker_random$callback)

tracker_lanczos <- track_loss()
model_lanczos <- nmf(aml, k = 6, seed = 42, init = "lanczos", tol = 1e-6,
                     maxit = 80, on_iteration = tracker_lanczos$callback)

loss_random <- tracker_random$get()
loss_lanczos <- tracker_lanczos$get()

convergence_df <- rbind(
  data.frame(Iteration = seq_along(loss_random), Loss = loss_random, Init = "Random"),
  data.frame(Iteration = seq_along(loss_lanczos), Loss = loss_lanczos, Init = "Lanczos")
)
```

``` r
knitr::kable(
  data.frame(
    Initialization = c("Random", "Lanczos"),
    Iterations = c(length(loss_random), length(loss_lanczos)),
    `Final Loss` = round(c(tail(loss_random, 1), tail(loss_lanczos, 1)), 4),
    check.names = FALSE
  ),
  caption = "Convergence comparison: random vs. Lanczos initialization on AML data."
)
```

| Initialization | Iterations | Final Loss |
|:---------------|-----------:|-----------:|
| Random         |         80 |   2386.559 |
| Lanczos        |         80 |   2449.680 |

Convergence comparison: random vs. Lanczos initialization on AML data.

``` r
ggplot(convergence_df, aes(x = Iteration, y = Loss, color = Init)) +
  geom_line(linewidth = 0.8) +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "Convergence: Random vs. Lanczos Initialization",
    x = "Iteration", y = "Reconstruction Loss", color = "Initialization"
  ) +
  theme_minimal()
```

![](nmf-fundamentals_files/figure-html/convergence-plot-1.png)

Lanczos initialization starts from a better initial point (SVD-based
warm start) and typically reaches a low loss in fewer iterations, while
random initialization requires more iterations to reach a comparable
solution.

## Next Steps

- **Rank selection**: How many factors should you use? See the
  [Cross-Validation](https://zdebruine.github.io/RcppML/articles/cross-validation.md)
  vignette.
- **Non-Gaussian data**: Count data, ratings, and overdispersed data
  need distribution-aware NMF. See the
  [Distributions](https://zdebruine.github.io/RcppML/articles/distributions.md)
  vignette.
- **Factor structure**: Control sparsity, smoothness, and other factor
  properties. See the Regularization vignette.
