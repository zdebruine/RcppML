# StreamPress I/O: Read, Write, and Inspect Compressed Matrices

Functions for reading and writing matrices in StreamPress (.spz) format.
StreamPress achieves 5-10x compression over raw float32 CSC binary on
typical scRNA-seq sparse matrices using rANS entropy coding. Beyond
storage savings, SPZ is also faster to read than raw CSC binary at any
thread count: the bottleneck in reading large sparse matrices is sparse
object construction (sorting indices, allocating R/Eigen structures),
which SPZ parallelises across independent chunks while raw CSC must
perform sequentially.

## Details

StreamPress (.spz) supports two format versions:

- **v2 (sparse)**: Column-oriented compressed CSC format. Lossless,
  5-10x compression over raw float32 CSC. Self-describing header.

- **v3 (dense)**: Column-major dense panels with optional
  FP16/QUANT8/rANS compression. For streaming NMF on dense data.

## See also

[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md),
[`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md),
[`st_write_dense`](https://zdebruine.github.io/RcppML/reference/st_write_dense.md),
[`st_read_dense`](https://zdebruine.github.io/RcppML/reference/st_read_dense.md)
