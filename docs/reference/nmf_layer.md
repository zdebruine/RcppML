# Create an NMF factorization layer

Creates a non-negative matrix factorization layer that decomposes its
input into W \* diag(d) \* H with non-negativity constraints.

## Usage

``` r
nmf_layer(
  input,
  k,
  L1 = 0,
  L2 = 0,
  L21 = 0,
  angular = 0,
  upper_bound = 0,
  mask = NULL,
  zi = c("none", "row", "col"),
  projective = FALSE,
  symmetric = FALSE,
  robust = FALSE,
  W = NULL,
  H = NULL,
  name = NULL,
  ...
)
```

## Arguments

- input:

  An `fn_node` (input, shared, condition, or another layer).

- k:

  Factorization rank.

- L1:

  L1 penalty (shared by W and H unless overridden). Default 0.

- L2:

  L2 penalty. Default 0.

- L21:

  Group sparsity penalty. Default 0.

- angular:

  Orthogonality penalty. Default 0.

- upper_bound:

  Box constraint. Default 0.

- mask:

  Masking mode: NULL (none), `"zeros"`, `"NA"`, or a sparse mask matrix.
  See [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) for
  details.

- zi:

  Zero-inflation mode: `"none"`, `"row"`, or `"col"`. Requires
  `loss = "gp"` or `"nb"` in
  [`factor_config()`](https://zdebruine.github.io/RcppML/reference/factor_config.md).
  Default `"none"`.

- projective:

  Use projective NMF (W is reused as H). Default FALSE.

- symmetric:

  Use symmetric NMF (W == H). Default FALSE.

- robust:

  Robustness control: `FALSE` (default, no robustness), `TRUE` (Huber
  delta=1.345), `"mae"` (near-MAE, delta=1e-4), or a positive numeric
  Huber delta. See
  [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) for
  details.

- W:

  Optional [`W()`](https://zdebruine.github.io/RcppML/reference/W.md)
  config to override W-specific settings.

- H:

  Optional [`H()`](https://zdebruine.github.io/RcppML/reference/W.md)
  config to override H-specific settings.

- name:

  Optional layer name (for results access).

- ...:

  Additional arguments forwarded to
  [`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) at fit
  time. Supports all advanced parameters: distribution tuning
  (`dispersion`, `theta_init`, etc.), IRLS control (`irls_max_iter`,
  `irls_tol`), solver tuning (`cd_tol`, `cd_maxit`), streaming
  (`streaming`, `panel_cols`), callbacks (`on_iteration`), and more. See
  [`?nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md) for the
  complete list.

## Value

An `fn_node` of type "nmf_layer".

## Details

When used with `|>`, the input node is the first argument:
`x |> nmf_layer(k = 64)`.

Layer-level regularization parameters (L1, L2, etc.) apply to both W and
H unless overridden by
[`W()`](https://zdebruine.github.io/RcppML/reference/W.md) or
[`H()`](https://zdebruine.github.io/RcppML/reference/W.md).

## See also

[`svd_layer`](https://zdebruine.github.io/RcppML/reference/svd_layer.md),
[`factor_net`](https://zdebruine.github.io/RcppML/reference/factor_net.md),
[`factor_input`](https://zdebruine.github.io/RcppML/reference/factor_input.md),
[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md)

## Examples

``` r
data(aml)
inp <- factor_input(aml)
layer <- nmf_layer(inp, k = 5)

# With zero-inflation and distribution-specific tuning
layer_gp <- nmf_layer(inp, k = 5, zi = "row",
                      dispersion = "per_col", theta_init = 0.5)
```
