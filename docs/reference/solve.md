# Solve a linear system with coordinate descent

Solves the equation `a %*% x = b` for `x`, optionally subject to \\x \>=
0\\.

## Usage

``` r
solve(
  a,
  b,
  cd_maxit = 100L,
  cd_tol = 1e-08,
  L1 = 0,
  L2 = 0,
  upper_bound = 0,
  nonneg = TRUE
)
```

## Arguments

- a:

  symmetric positive definite matrix giving coefficients of the linear
  system

- b:

  matrix giving the right-hand side(s) of the linear system

- cd_maxit:

  maximum number of coordinate descent iterations

- cd_tol:

  stopping criteria, difference in \\x\\ across consecutive solutions
  over the sum of \\x\\

- L1:

  L1/LASSO penalty to be subtracted from `b`

- L2:

  Ridge penalty to be added to the diagonal of `a`

- upper_bound:

  maximum value permitted in solution, set to `0` to impose no upper
  bound

- nonneg:

  if TRUE (default), impose non-negativity constraint on solution

## Value

vector or matrix giving solution for `x`

## Details

This is a very fast implementation of sequential coordinate descent
least squares, suitable for very small or very large systems. The
algorithm begins with a zero-filled initialization of `x`.

Least squares by **sequential coordinate descent** is used to ensure the
solution returned is exact. This algorithm was introduced by Franc et
al. (2005), and our implementation is a vectorized and optimized
rendition of that found in the NNLM R package by Xihui Lin (2020).

When `nonneg = TRUE`, this solves the Non-Negative Least Squares (NNLS)
problem. When `nonneg = FALSE`, this solves unconstrained least squares.

## Note

This function is deprecated. Use
[`nnls`](https://zdebruine.github.io/RcppML/reference/nnls.md) instead.
`solve()` previously masked
[`base::solve()`](https://rdrr.io/r/base/solve.html).

## References

DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance
non-negative matrix factorization for large single-cell data." BioRXiv.

Franc, VC, Hlavac, VC, and Navara, M. (2005). "Sequential
Coordinate-Wise Algorithm for the Non-negative Least Squares Problem.
Proc. Int'l Conf. Computer Analysis of Images and Patterns."

Lin, X, and Boutros, PC (2020). "Optimization and expansion of
non-negative matrix factorization." BMC Bioinformatics.

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`nnls`](https://zdebruine.github.io/RcppML/reference/nnls.md)

## Author

Zach DeBruine

## Examples

``` r
if (FALSE) { # \dontrun{
# compare solution to base::solve for a random system
X <- matrix(runif(100), 10, 10)
a <- crossprod(X)
b <- crossprod(X, runif(10))

# unconstrained solution (same as base::solve)
unconstrained_soln <- solve(a, b, nonneg = FALSE)

# non-negative constrained solution
nonneg_soln <- solve(a, b, nonneg = TRUE)
} # }
```
