# Concatenate H factors from branches (row-bind)

Creates a node that row-binds the H factors from multiple branches.
Allows combining branches with different ranks into a single
higher-dimensional representation: k_out = k_1 + k_2 + ...

## Usage

``` r
factor_concat(...)
```

## Arguments

- ...:

  Two or more `fn_node` objects (layer outputs).

## Value

An `fn_node` of type "concat".

## Examples

``` r
data(aml)
inp <- factor_input(aml)
branch1 <- nmf_layer(inp, k = 3)
branch2 <- nmf_layer(inp, k = 5)
combined <- factor_concat(branch1, branch2)
```
