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
if (FALSE) { # \dontrun{
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
print(cv)
cv$best_params  # optimal rank

# Multi-parameter search
cv2 <- cross_validate_graph(
  inputs = inp,
  layer_fn = function(p) inp |> nmf_layer(k = p$k, L1 = p$L1),
  params = list(k = c(5, 10, 20), L1 = c(0, 0.01, 0.1)),
  config = factor_config(maxit = 50, seed = 42),
  reps = 3
)
} # }
```
