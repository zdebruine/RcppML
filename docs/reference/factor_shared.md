# Shared factorization across multiple inputs (multi-modal)

Creates a node representing shared-H factorization of multiple input
matrices. The resulting H is shared across all inputs; each input gets
its own W.

## Usage

``` r
factor_shared(...)
```

## Arguments

- ...:

  Two or more `fn_node` objects (inputs or layers).

## Value

An `fn_node` of type "shared".

## Details

Execution concatenates inputs row-wise and runs a single NMF:
`rbind(X1, X2, ...) = rbind(W1, W2, ...) * diag(d) * H`.

## Examples

``` r
data(aml)
# Split into two views
inp1 <- factor_input(aml[1:400, ], name = "view1")
inp2 <- factor_input(aml[401:824, ], name = "view2")
shared <- factor_shared(inp1, inp2)
```
