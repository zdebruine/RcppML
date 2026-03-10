# RcppML: Fast Non-Negative Matrix Factorization and Divisive Clustering

High-performance non-negative matrix factorization (NMF), singular value
decomposition (SVD), and divisive clustering for large sparse and dense
matrices. Implements alternating least squares with coordinate descent
and Cholesky NNLS solvers, diagonal scaling for interpretable factors,
cross-validation for automatic rank selection, multiple
distribution-based losses (Gaussian, Poisson, Generalized Poisson,
Negative Binomial, Gamma, Inverse Gaussian, Tweedie) via iteratively
reweighted least squares, regularization (L1, L2, L21, angular, graph
Laplacian), and optional GPU acceleration via CUDA. Includes divisive
clustering via recursive rank-2 factorization, consensus clustering, and
the SparsePress compressed sparse matrix format. Methods are described
in DeBruine, Melcher, and Triche (2021)
[doi:10.1101/2021.09.01.458620](https://doi.org/10.1101/2021.09.01.458620)
.

## See also

Useful links:

- <https://github.com/zdebruine/RcppML>

- Report bugs at <https://github.com/zdebruine/RcppML/issues>

## Author

**Maintainer**: Zachary DeBruine <zacharydebruine@gmail.com>
([ORCID](https://orcid.org/0000-0003-2234-4827))
