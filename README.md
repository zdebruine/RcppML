# RcppML

An R package for high-performance linear model projection, non-negative least squares, L0-regularized least squares, and more.

## Why RcppML?
* Simple and well-documented suite of R and C++ functions
* **Non-negative least squares**: Fastest NNLS solver
* **Sparse matrix support**: Any sparse matrix that fits in memory can be used. Specialized support for in-place operations.
* **L0 regularization**: computationally tractable near-exact and approximate methods
* **Fast**: Everything is exhaustively microbenchmarked and written in templated C++ using the Eigen library.

## R package
Install RcppML:
```{R}
library(devtools)
install_github("zdebruine/RcppML")
```

## Documentation
* Exhaustive documentation, examples, benchmarking, and developer guide in the bookdown website
* Get started with the package vignette

## C++ Header Library
* Most `RcppML` functions are simple wrappers of the Eigen header library contained in the `inst/include` directory.
* Functions in this header library are separately documented and may be used in C++ applications.

## Solving linear systems
Unconstrained or non-negative solutions to linear systems are found quickly by LLT decomposition/substitution followed by refinement by coordinate descent. Prior to solving, regularizations may also be applied.

## Active development
RcppML is under non-breaking active development. Functionality to be released in the next few months will build off the current library and includes:
* Unconstrained or non-negative diagonalized matrix factorization by alternating least squares with convex L1 regularization
* Efficient and naturally robust solutions to large matrix factorizations
* Extremely fast rank-1 factorization
* Extremely fast exact rank-2 matrix factorizations (faster than _irlba_ rank-2 SVD)
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations

### RcppML::CoeffMatrix class

**Definition**

The `CoeffMatrix` class makes use of metaprogramming templates for compile-time optimization:

```{Cpp}
template <typename T, bool nonneg = false, int SizeAtCompileTime = -1, double tol = 1e-8, int maxit = 100, int L0 = -1, std::string L0_path = "ecd">
class CoeffMatrix {
public:
    Eigen::Matrix<T, SizeAtCompileTime, SizeAtCompileTime> a;
    Eigen::LLT <Eigen::Matrix<T, SizeAtCompileTime, SizeAtCompileTime>> a_llt;
}
```
* _SizeAtCompileTime_: `-1` indicates that the size of a default coefficient matrix `a` is dynamic. There is a special solver for 2x2 matrices, and Eigen can optimize computation for other small fixed sizes
* _nonneg_: Apply non-negativity constraints
* _tol_: tolerance of coordinate descent refinement
* _maxit_: maximum number of refinement iterations in coordinate descent
* _L0_: maximum cardinality of the solution to be returnd (default 0 for no regularization)
* _L0_path_: algorithm for finding the L0 solution, one of "exact", "appx", "ecd", or "convex".


**Constructor**

_CoeffMatrix(a, L1, L2, iL2)_
* _a_ is a symmetric positive definite matrix giving the coefficients of the linear system
* _L1_ gives the Lasso regularization to be subtracted from _b_ (default 0)
* _L2_ gives the Ridge regularization to be added to the diagonal elements of _a_ (default 0)
* _iL2_ (inverse L2) gives the "angular" or "pattern extraction" regularization to be added to the off-diagonal elements of _a_ (default 0)

At the time of construction, a reference to _a_ is stored in the class, _L2_ and _iL2_ regularizations are applied, and the _.llt()_ decomposition is computed and stored. _L1_ regularization is applied to _b_ when _.solve()_ is called.


**Member functions**

There are only two member functions:
* _.solve(b)_ returns an object of the same class as _b_, where _b_ is a vector or column/row subview of the same length as the edge length of `a`.
* _.solveInPlace(b)_ updates _b_ with the solution, and is only faster than _.solve(b)_ for 2-variable solutions.


**Example**

```{Cpp}
Matrix3f  mat; mat << 1, 2, 3,
                      2, 4, 5,
                      3, 5, 6;
Vector3f vec;  vec << 1, 2, 3;

RcppML::CoeffMatrix<mat, true, 3> a(mat);
Vector3f x = a.solve(b);
```
