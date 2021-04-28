# fastlm

An R package for high-performance linear model projection, non-negative least squares, L0-regularized least squares, and more.

**Why fastlm?**
* Fastest non-negative least squares solver by far
* Exact and approximate L0-regularized least squares solutions
* Support for extremely large sparse matrices
* Exhaustively microbenchmarked templated C++ implementations using very fast Eigen library solvers

**R package**
Install fastlm:
```{R}
library(devtools)
install_github("zdebruine/fastlm")
```

The public development branch `zdebruine/fastlm-dev` is unstable and undocumented. 

**Documentation**
* Exhaustive documentation, examples, benchmarking, and developer guide in the pkgdown website
* Get started with the package vignette

**C++ Header Library**
* Most `fastlm` functions are simple wrappers of the Eigen header library contained in the `inst/include` directory.
* Functions in this header library are separately documented and may be used in C++ applications.

**Ongoing work**:
fastlm is under non-breaking active development. Functionality to be released in the next few months will build off the current library and includes:
* Unconstrained or non-negative diagonalized matrix factorization by alternating least squares with convex L1 regularization
* Efficient and naturally robust solutions to large matrix factorizations
* Extremely fast rank-1 factorization
* Extremely fast exact rank-2 matrix factorizations (faster than _irlba_ rank-2 SVD)
* Divisive clustering using recursive bipartitioning by rank-2 matrix factorizations
