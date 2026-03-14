# Cross-validate a factorization network

Evaluates a factor_net architecture across a grid of hyperparameter
combinations using held-out test loss. Each combination is fitted with a
fraction of entries masked, and the test loss on those entries
determines the optimal configuration.

## Usage

``` r
cross_validate_graph(
  inputs,
  layer_fn,
  params,
  config = factor_config(),
  reps = 3L,
  strategy = c("grid", "random"),
  n_random = 20L,
  seed = 42L,
  verbose = TRUE
)
```

## Arguments

- inputs:

  Input node(s) from
  [`factor_input()`](https://zdebruine.github.io/RcppML/reference/factor_input.md).
  Can be a single node or a list of nodes for multi-modal.

- layer_fn:

  A function that, given a named list of parameter values, returns the
  output layer node of the network. Example:
  `function(p) inputs |> nmf_layer(k = p$k, L1 = p$L1)`

- params:

  A named list of parameter vectors to search over. Names should match
  the arguments expected by `layer_fn`. Example:
  `list(k = c(5, 10, 20), L1 = c(0, 0.01))`.

- config:

  A `fn_global_config` from
  [`factor_config()`](https://zdebruine.github.io/RcppML/reference/factor_config.md).
  The `test_fraction`, `cv_seed`, `mask_zeros`, and `patience` fields
  are used for cross-validation. If `test_fraction` is 0 (default), it
  is automatically set to 0.1.

- reps:

  Number of CV replicates per parameter combination (each with a
  different CV mask seed). Default 3.

- strategy:

  Search strategy: `"grid"` (all combinations) or `"random"` (sample
  `n_random` combinations). Default "grid".

- n_random:

  Number of combinations for random search. Ignored for grid search.
  Default 20.

- seed:

  Seed for random search sampling and CV mask derivation. Default 42.

- verbose:

  Print progress updates. Default TRUE.

## Value

A `factor_net_cv` object with components:

- results:

  Data frame with columns: param values, rep, test_loss, train_loss,
  iterations, converged.

- summary:

  Data frame with param values, mean_test_loss, se_test_loss,
  mean_train_loss, ranked by mean_test_loss.

- best_params:

  Named list of the best parameter combination.

- all_fits:

  List of all fit results (if `keep_fits = TRUE`).

## Details

For single-parameter rank selection, pass `k = c(5, 10, 20)`. For
multi-parameter search, use `params` to specify named lists of values
for each layer and parameter.

## Note

The `seed` parameter defaults to `42L` (deterministic) rather than
`NULL` (random) used by
[`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) and
[`svd()`](https://zdebruine.github.io/RcppML/reference/svd.md). This
ensures reproducible cross-validation grid searches by default. Pass
`seed = NULL` for non-deterministic behavior.

## See also

[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md),
[`fit`](https://zdebruine.github.io/RcppML/reference/fit.md),
[`factor_config`](https://zdebruine.github.io/RcppML/reference/factor_config.md)

## Examples

``` r
# \donttest{
library(Matrix)
X <- rsparsematrix(100, 50, 0.1)
inp <- factor_input(X, "X")

# Rank selection
cv <- cross_validate_graph(
  inputs = inp,
  layer_fn = function(p) inp |> nmf_layer(k = p$k),
  params = list(k = c(3, 5, 10, 20)),
  config = factor_config(maxit = 50, seed = 42)
)
#> Cross-validating 4 parameter combinations x 3 reps = 12 fits
#>   [1/4] k = 3
#>   [2/4] k = 5
#>   [3/4] k = 10
#>   [4/4] k = 20
#> 
#> Best: k = 3 -> test_loss = 0.301088 (SE = 0.046311)
print(cv)
#> factor_net cross-validation
#>   Strategy: grid | Reps: 3 | Combos: 4
#>   Holdout: 10.0%
#> 
#> Ranked results (by mean test loss):
#>   k mean_test_loss se_test_loss mean_train_loss n_valid
#>   3      0.3010881   0.04631067      0.09013363       3
#>  10      0.4014847   0.07288210      0.07651645       3
#>   5      0.4480407   0.10936952      0.08320824       3
#>  20      1.1561439   0.12199328      0.07239668       3
#> 
#> Best: k = 3
cv$best_params  # optimal rank
#> $k
#> [1] 3
#> 

# Multi-parameter search
cv2 <- cross_validate_graph(
  inputs = inp,
  layer_fn = function(p) inp |> nmf_layer(k = p$k, L1 = p$L1),
  params = list(k = c(5, 10, 20), L1 = c(0, 0.01, 0.1)),
  config = factor_config(maxit = 50, seed = 42),
  reps = 3
)
#> Cross-validating 9 parameter combinations x 3 reps = 27 fits
#>   [1/9] k = 5, L1 = 0
#>   [2/9] k = 10, L1 = 0
#>   [3/9] k = 20, L1 = 0
#>   [4/9] k = 5, L1 = 0.01
#>   [5/9] k = 10, L1 = 0.01
#>   [6/9] k = 20, L1 = 0.01
#>   [7/9] k = 5, L1 = 0.1
#>   [8/9] k = 10, L1 = 0.1
#>   [9/9] k = 20, L1 = 0.1
#> 
#> Best: k = 5, L1 = 0.1 -> test_loss = 0.197064 (SE = 0.014186)
# }
```
