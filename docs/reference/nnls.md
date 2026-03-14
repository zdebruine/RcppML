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
  L1 = c(0, 0),
  L2 = c(0, 0),
  loss = "mse",
  upper_bound = c(0, 0),
  nonneg = c(TRUE, TRUE),
  threads = 0,
  verbose = FALSE,
  ...
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

  data matrix. Can be dense (matrix) or sparse (dgCMatrix).

- L1:

  L1/LASSO penalty, length 1 or 2. Range \[0, 1).

- L2:

  Ridge penalty, length 1 or 2. Range \[0, Inf).

- loss:

  loss function: `"mse"` (default) or others via NMF dispatch.

- upper_bound:

  maximum value in solution, length 1 or 2. 0 = no bound.

- nonneg:

  non-negativity constraints, length 1 or 2.

- threads:

  number of threads for OpenMP parallelization (0 = all available).

- verbose:

  print progress information.

- ...:

  advanced parameters. See **Advanced Parameters** section.

## Value

matrix of dimension (k x n_samples) for H or (n_features x k) for W

## Details

This function solves NNLS projection problems with full flexibility.

**Projection Modes:**

- `w` provided, `h = NULL`: Solve for H given W and A (standard
  projection)

- `h` provided, `w = NULL`: Solve for W given H and A (transpose
  projection)

- Exactly one of `w` or `h` must be NULL

## Advanced Parameters (via `...`)

- `L21`:

  L2,1 group sparsity penalty, length 1 or 2 (default `c(0,0)`)

- `angular`:

  Angular decorrelation penalty, length 1 or 2 (default `c(0,0)`)

- `cd_maxit`:

  Max coordinate descent iterations (default 100)

- `cd_tol`:

  CD stopping tolerance (default 1e-8)

- `warm_start`:

  Optional initial solution matrix

- `dispersion`:

  Dispersion estimation mode: `"per_row"` (default) or `"global"`

- `theta_init`:

  Initial GP theta (default 0.1)

- `theta_max`:

  Maximum GP theta (default 5.0)

- `theta_min`:

  Minimum GP theta (default 0.0)

- `nb_size_init`:

  Initial NB size parameter (default 10.0)

- `nb_size_max`:

  Maximum NB size (default 1e6)

- `nb_size_min`:

  Minimum NB size (default 0.01)

- `gamma_phi_init`:

  Initial Gamma shape parameter (default 1.0)

- `gamma_phi_max`:

  Maximum Gamma shape (default 1e4)

- `gamma_phi_min`:

  Minimum Gamma shape (default 1e-6)

- `tweedie_power`:

  Tweedie variance power (default 1.5)

- `irls_max_iter`:

  Maximum IRLS iterations per solve (default 5)

- `irls_tol`:

  IRLS convergence tolerance (default 1e-4)

## Target Regularization

When `target_H` and `target_lambda` are provided (via `...`), `nnls()`
routes the solve through a single NMF iteration internally. Positive
`target_lambda` attracts the solution toward `target_H` (label
enrichment). Negative `target_lambda` uses eigenvalue-projected
adversarial removal (PROJ_ADV) to suppress target-correlated structure
(batch removal). See
[`vignette("guided-nmf")`](https://zdebruine.github.io/RcppML/articles/guided-nmf.md)
for details.

## References

DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance
non-negative matrix factorization for large single-cell data." BioRXiv.

Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential
Coordinate-Wise Algorithm for the Non-negative Least Squares Problem."

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

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

# With L1 penalty for sparse H
h_sparse <- nnls(w = w, A = A, L1 = c(0, 0.1))
# }
```
