# Write a dense matrix to an SPZ v3 file

Write a dense matrix to an SPZ v3 file

## Usage

``` r
Rcpp_sp_write_dense(
  A,
  path,
  include_transpose = FALSE,
  chunk_cols = 256L,
  codec = 0L,
  delta = FALSE
)
```

## Arguments

- A:

  A numeric matrix (dense)

- path:

  Output file path (.spz)

- include_transpose:

  Also store transposed panels for streaming NMF

- chunk_cols:

  Columns per chunk (default 256)

- codec:

  Compression codec: 0=RAW_FP32, 1=FP16, 2=QUANT8, 3=FP16_RANS,
  4=FP32_RANS

- delta:

  Apply XOR-delta encoding before entropy coding

## Value

A list with write statistics
