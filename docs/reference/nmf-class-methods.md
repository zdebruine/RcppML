# nmf class methods

Given an NMF model in the form \\A = wdh\\, `project` projects `w` onto
`A` to solve for `h`.

## Usage

``` r
# S4 method for class 'nmf'
subset(x, i, ...)

# S4 method for class 'nmf,ANY,ANY,ANY'
x[i]

# S4 method for class 'nmf'
head(x, n = getOption("digits"), ...)

# S4 method for class 'nmf'
show(object)

# S4 method for class 'nmf'
dimnames(x)

# S4 method for class 'nmf'
dim(x)

# S4 method for class 'nmf'
t(x)

# S4 method for class 'nmf'
sort(x, decreasing = TRUE, ...)

# S4 method for class 'nmf'
prod(x, ..., na.rm = FALSE)

# S4 method for class 'nmf'
x$name

# S4 method for class 'nmf,list'
coerce(from, to)

# S4 method for class 'nmf'
x[[i]]

# S4 method for class 'nmf'
predict(
  object,
  data,
  L1 = NULL,
  L2 = NULL,
  mask = NULL,
  upper_bound = NULL,
  threads = 0,
  verbose = FALSE,
  ...
)
```

## Arguments

- x:

  object of class `nmf`.

- i:

  indices

- ...:

  arguments passed to or from other methods

- n:

  number of rows/columns to show

- object:

  fitted model, class `nmf`, generally the result of calling `nmf`, with
  models of equal dimensions as `data`

- decreasing:

  logical. Should the sort be increasing or decreasing?

- na.rm:

  remove na values

- name:

  name of nmf class slot

- from:

  class which the coerce method should perform coercion from

- to:

  class which the coerce method should perform coercion to

- data:

  dense or sparse matrix of features in rows and samples in columns.
  Prefer `matrix` or `Matrix::dgCMatrix`, respectively. Also accepts a
  file path (character string) which will be auto-loaded based on
  extension.

- L1:

  a single LASSO penalty in the range (0, 1\]

- L2:

  a single Ridge penalty greater than zero

- mask:

  missing data mask. Accepts: `NULL` (no masking), `"zeros"` (mask
  zeros), `"NA"` (mask NAs), a dgCMatrix/matrix (custom mask), or
  `list("zeros", <matrix>)` to mask zeros and use a custom mask
  simultaneously.

- upper_bound:

  maximum value permitted in least squares solutions, essentially a
  bounded-variable least squares problem between 0 and `upper_bound`

- threads:

  number of threads for OpenMP parallelization (default 0 = all
  available)

- verbose:

  print progress information (default FALSE)

## Value

An `nmf` object subsetted to the specified factors.

An `nmf` object subsetted to the specified factors.

Invisibly returns the `nmf` object.

Invisibly returns the `nmf` object.

A list of length two: row names of `w` and column names of `h`.

An integer vector of length 3: number of features, number of samples,
and rank.

An `nmf` object with transposed factorization (`w` and `h` swapped and
transposed).

An `nmf` object with factors reordered by decreasing (or increasing)
`d`.

A dense matrix equal to `w %*% diag(d) %*% h`.

Contents of the named slot (`w`, `d`, `h`, or `misc`), or the named
element from `misc`.

A named list with elements `w`, `d`, `h`, and `misc`.

The element from the `misc` list at the given name or index.

An [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) object
containing the projected model (with updated `h` matrix).

## Details

For the alternating least squares matrix factorization update problem
\\A = wh\\, the updates (or projection) of \\h\\ is given by the
equation: \$\$w^Twh = wA_j\$\$ which is in the form \\ax = b\\ where \\a
= w^Tw\\ \\x = h\\ and \\b = wA_j\\ for all columns \\j\\ in \\A\\.

Any L1 penalty is subtracted from \\b\\ and should generally be scaled
to `max(b)`, where \\b = WA_j\\ for all columns \\j\\ in \\A\\. An easy
way to properly scale an L1 penalty is to normalize all columns in \\w\\
to sum to the same value (e.g. 1). No scaling is applied in this
function. Such scaling guarantees that `L1 = 1` gives a completely
sparse solution.

There are specializations for dense and sparse input matrices, symmetric
input matrices, and for rank-1 and rank-2 projections. See documentation
for [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) for
theoretical details and guidance.

## References

DeBruine, ZJ, Melcher, K, and Triche, TJ. (2021). "High-performance
non-negative matrix factorization for large single-cell data." BioRXiv.

## See also

[`nnls`](https://zdebruine.github.io/RcppML/reference/nnls.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Author

Zach DeBruine

## Examples

``` r
# \donttest{
library(Matrix)
w <- matrix(runif(1000 * 10), 1000, 10)
h_true <- matrix(runif(10 * 100), 10, 100)
# A is the crossproduct of "w" and "h" with 10% signal dropout
mask <- rsparsematrix(1000, 100, density = 0.1)
A <- (w %*% h_true) * (mask != 0)
h <- nnls(w = w, A = A)
cor(as.vector(h_true), as.vector(h))
#> [1] 0.3003039

# alternating projections refine solution (like NMF)
h <- nnls(w = w, A = A)
w <- nnls(h = h, A = A)
h <- nnls(w = w, A = A)
w <- nnls(h = h, A = A)
h <- nnls(w = w, A = A)
w <- nnls(h = h, A = A)
# }
```
