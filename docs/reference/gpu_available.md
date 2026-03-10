# Check if GPU acceleration is available

Detects CUDA GPUs at runtime. Result is cached for the session.

## Usage

``` r
gpu_available(force_recheck = FALSE)
```

## Arguments

- force_recheck:

  Re-probe GPUs even if already cached

## Value

logical TRUE if GPU is available

## Examples

``` r
gpu_available()
#> [1] FALSE
```
