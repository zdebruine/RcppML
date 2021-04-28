# fastlm

An R package for high-performance linear models.

**Why fastlm?**
* Fastest non-negative least squares solver by far
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
