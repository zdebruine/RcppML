# Contributing to RcppML

Thank you for your interest in contributing to RcppML! This document provides
guidelines and instructions for contributing to the project.

## Getting Started

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/zdebruine/RcppML.git
   cd RcppML
   ```

2. **Install dependencies**:
   ```r
   install.packages(c("Rcpp", "RcppEigen", "Matrix", "testthat", "devtools", "roxygen2", "knitr", "rmarkdown"))
   ```

3. **Build and test**:
   ```r
   devtools::document()
   devtools::test()
   devtools::check()
   ```

## Project Structure

| Directory | Contents |
|-----------|----------|
| `R/` | R source files (roxygen documented) |
| `src/` | C++ Rcpp bridge (`RcppFunctions.cpp`) |
| `inst/include/RcppML/` | Header-only C++ library (core algorithms) |
| `tests/testthat/` | Unit tests (testthat) |
| `vignettes/` | Package vignettes |
| `man/` | **Auto-generated** — do NOT edit |
| `NAMESPACE` | **Auto-generated** — do NOT edit |

## Development Workflow

### Editing R Code

1. Edit files in `R/` with roxygen2 comments (`#' @param`, `#' @export`, etc.)
2. Regenerate documentation:
   ```r
   devtools::document()
   ```
3. Fix the Rcpp info bug:
   ```bash
   bash tools/fix_rcpp_info_bug.sh
   ```
4. Run tests:
   ```r
   devtools::test()
   ```

### Editing C++ Code

All C++ algorithm code lives in `inst/include/RcppML/` as a header-only library.
The only compiled file is `src/RcppFunctions.cpp`, which provides `// [[Rcpp::export]]`
wrappers.

After changing `// [[Rcpp::export]]` annotations, run:
```r
devtools::document()  # Also runs Rcpp::compileAttributes()
```
Then:
```bash
bash tools/fix_rcpp_info_bug.sh
```

### Auto-Generated Files (NEVER Edit Manually)

| File(s) | Generator |
|---------|-----------|
| `NAMESPACE` | `roxygen2` via `devtools::document()` |
| `man/*.Rd` | `roxygen2` via `devtools::document()` |
| `src/RcppExports.cpp` | `Rcpp::compileAttributes()` via `devtools::document()` |
| `R/RcppExports.R` | `Rcpp::compileAttributes()` via `devtools::document()` |

## Code Style

### R Code
- Follow standard R style (snake_case for functions, PascalCase for S4 classes)
- All exported functions must have complete roxygen documentation including `@return` and `@examples`
- Use `stop()` for user-facing errors with clear messages

### C++ Code
- Namespace: `RcppML` (PascalCase) for all new code
- Templates: use `Scalar` as the floating-point type parameter
- Headers must be Rcpp-free — use `RCPPML_LOG_INFO()` instead of `Rcpp::Rcout`
- Sparse matrices: expect `Eigen::SparseMatrix<Scalar>` in CSC format
- Internal W is stored transposed (k × m) for cache efficiency

## Testing

### Running Tests
```r
devtools::test()                                    # All tests
testthat::test_file("tests/testthat/test_nmf.R")    # Single file
```

### Writing Tests
- Place test files in `tests/testthat/` named `test_*.R`
- Use `skip_if_not_installed()` for optional dependencies
- Use `skip_if(Sys.getenv("RCPPML_SKIP_GPU") == "true")` for GPU tests
- Ground truth recovery tests should use multi-restart for robustness

### GPU Tests
GPU tests require a CUDA-capable GPU and the compiled GPU library.

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes with tests
3. Ensure `devtools::check()` passes with no errors
4. Push and open a Pull Request against `main`

## Reporting Issues

Open an issue at https://github.com/zdebruine/RcppML/issues with:
- A minimal reproducible example
- R session info (`sessionInfo()`)
- Expected vs actual behavior
