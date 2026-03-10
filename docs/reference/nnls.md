# Non-negative Least Squares Projection

Project a factor matrix onto new data to solve for the complementary
matrix. Given \\W\\ and \\A\\, find \\H\\ such that \\\|\|WH - A\|\|^2\\
is minimized. Alternatively, given \\H\\ and \\A\\, find \\W\\ such that
\\\|\|WH - A\|\|^2\\ is minimized.

## Usage

``` r
nnls(
  w = NULL,
  h = NULL,
  A,
  warm_start = NULL,
  L1 = c(0, 0),
  L2 = c(0, 0),
  cd_maxit = 100L,
  cd_tol = 1e-08,
  upper_bound = c(0, 0),
  nonneg = c(TRUE, TRUE),
  threads = 0,
  verbose = FALSE
)
```

## Arguments

- w:

  factor model matrix (n_features x k) for solving H. Set to NULL to
  solve for W.

- h:

  factor model matrix (k x n_samples) for solving W. Set to NULL to
  solve for H.

- A:

  data matrix (n_features x n_samples for w-\>h, n_samples x n_features
  for h-\>w). Can be dense (matrix) or sparse (dgCMatrix from Matrix
  package).

- warm_start:

  optional matrix to initialize the solution. Must match output
  dimensions (k x n_samples for H, n_features x k for W). Can accelerate
  convergence.

- L1:

  L1/LASSO penalty, length 1 or 2 for c(w_penalty, h_penalty). Range
  \[0, 1).

- L2:

  Ridge penalty, length 1 or 2 for c(w_penalty, h_penalty). Range \[0,
  Inf).

- cd_maxit:

  maximum number of coordinate descent iterations (default 100)

- cd_tol:

  stopping tolerance for coordinate descent (default 1e-8)

- upper_bound:

  maximum value in solution, length 1 or 2 for c(w_bound, h_bound). 0 =
  no bound.

- nonneg:

  non-negativity constraints, length 1 or 2 for c(w_constraint,
  h_constraint)

- threads:

  number of threads for OpenMP parallelization (0 = all available)

- verbose:

  print progress information

## Value

matrix of dimension (k x n_samples) for H or (n_features x k) for W

## Details

This function solves NNLS projection problems with full flexibility:

**Projection Modes:**

- `w` provided, `h = NULL`: Solve for H given W and A (standard
  projection)

- `h` provided, `w = NULL`: Solve for W given H and A (transpose
  projection)

- Exactly one of `w` or `h` must be NULL

**Normal Equations:**

- For H: \\W'W H = W'A\\

- For W: \\HH' W' = HA'\\ (solved as transpose problem)

The problem is solved by forming the Gram matrix and right-hand side in
C++ with OpenMP parallelization across samples. Sequential coordinate
descent with warm starts enables fast convergence.

**Dimension Matching:** When dimensions don't match exactly, the
function attempts to:

1.  Auto-transpose if one orientation matches

2.  Match by rownames/colnames and reorder/subset to the intersection

3.  Error if no valid dimension alignment is possible

**Warm Start:** Provide an optional `warm_start` matrix to initialize
the solution. This can dramatically accelerate convergence when
solutions are expected to be similar across related problems (e.g.,
incremental updates, time series).

**Penalties:**

- L1 (LASSO): Encourages sparsity by subtracting L1 from the right-hand
  side

- L2 (Ridge): Shrinks solutions by adding L2 to the diagonal of the Gram
  matrix

- Graph Laplacian: *Not yet implemented in nnls. Use nmf() for graph
  regularization.*

- Length-2 vectors `c(w_penalty, h_penalty)` control penalties for W and
  H separately

## Note

`nnls()` uses `A` for the data matrix (linear algebra convention),
matching [`svd()`](https://zdebruine.github.io/RcppML/reference/svd.md)
and the normal equations \\W'WH = W'A\\. In contrast,
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) uses
`data` (statistical modeling convention).

## References

DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance
non-negative matrix factorization for large single-cell data." BioRXiv.

Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential
Coordinate-Wise Algorithm for the Non-negative Least Squares Problem."

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`solve`](https://zdebruine.github.io/RcppML/reference/solve.md)

## Author

Zach DeBruine

## Examples

``` r
# \donttest{
# Generate synthetic NMF problem
set.seed(123)
w <- matrix(runif(100), 20, 5)  # 20 features, 5 factors
h_true <- matrix(runif(50), 5, 10)  # 5 factors, 10 samples
A <- w %*% h_true + matrix(rnorm(200, 0, 0.1), 20, 10)  # Add noise

# Project W onto new data to find H
h_recovered <- nnls(w = w, A = A)
cor(as.vector(h_true), as.vector(h_recovered))
#> [1] 0.9668001

# Project H onto new features to find W (transpose problem)
w_recovered <- nnls(h = h_true, A = A)
cor(as.vector(w), as.vector(w_recovered))
#> [1] 0.8989978

# With L1 penalty for sparse H
h_sparse <- nnls(w = w, A = A, L1 = c(0, 0.1))

# Warm start for faster convergence
h_init <- matrix(0.5, 5, 10)
h_warm <- nnls(w = w, A = A, warm_start = h_init)

# Allow negative values (semi-NMF)
h_unconstrained <- nnls(w = w, A = A, nonneg = c(TRUE, FALSE))

# Dimension matching by names
rownames(w) <- paste0("gene_", 1:20)
rownames(A) <- paste0("gene_", 20:1)  # Reversed order
h_matched <- nnls(w = w, A = A)  # Automatically reorders A by rownames
# }
```
