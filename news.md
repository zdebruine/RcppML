## RcppML 0.5.0

### Major changes

- Launch pkgdown site
- Added the `nmf` S3 class to the result of `nmf` function
- Introduce S3 methods for NMF (`[`, `align`, `biplot`, `dim`, `dimnames`, `head`, `mse`, `predict`, `print`, `prod`, `sort`, `sparsity`, `summary`, `t`)
- New plotting methods for NMF (`biplot.nmf`, `plot.nmfSummary`, `plot.nmfCrossValidation`)
- `mse` is now an S3 method for `nmf` objects
- `project` now handles only projections of `w`, for simplicity
- New vignette on `Getting Started with NMF`!

### Minor changes
- Support for specific sample and feature selections for NMF removed to increase performance on C++ end
- Removed `updateInPlace` advanced parameter for `nmf` because advantages were not convincing
- `mask_zeros` implementation is now specific to sparse matrices, multi-thread parallelization, and projections with transposition
- Added `cosine` function for fast cosine distance calculations
- Condensed and pared down documentation throughout. Advanced usage discussion will be moved to future vignettes.

## RcppML 0.5.1

### Major changes
- three new datasets (`hawaiibirds`, `aml`, and `movielens`)
- Move NMF models and methods from S3 to S4 for stability
- Better random initializations (now using both `rnorm` and `runif` with multiple ranges/shapes, when multiple seeds are specified)
- added L2 regularization to NMF
- Support for masking values
- add `impute` and `perturb` methods to `crossValidate`

### Minor changes
- better random initializations (now using both `rnorm` and `runif` with multiple ranges/shapes)
- New vignette on random restarts
- better "head" and "show" methods
- return "w_init" with model