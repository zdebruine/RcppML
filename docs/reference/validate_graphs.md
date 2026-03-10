# Validate graphs regularization matrices

Validate graphs regularization matrices

## Usage

``` r
validate_graphs(graph_W, graph_H, data_dims, graph_lambda)
```

## Arguments

- graph_W, graph_H:

  Optional sparse matrices for graph regularization

- data_dims:

  Integer vector `c(nrow, ncol)` of the input data

- graph_lambda:

  Length-2 numeric vector of graph lambda values

## Value

List with validated `graph_W` and `graph_H` (coerced to dgCMatrix)
