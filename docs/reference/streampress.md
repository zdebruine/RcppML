# StreamPress I/O: Read, Write, and Inspect Compressed Matrices

Functions for reading and writing matrices in StreamPress (.spz) format.
StreamPress (formerly SparsePress) achieves 10-20x compression on
typical scRNA-seq sparse matrices using rANS entropy coding, and
supports dense v3 format with multiple compression codecs.

## Details

StreamPress (.spz) supports two format versions:

- **v2 (sparse)**: Column-oriented compressed CSC format. Lossless,
  10-20x compression. Self-describing header.

- **v3 (dense)**: Column-major dense panels with optional
  FP16/QUANT8/rANS compression. For streaming NMF on dense data.

## See also

[`st_write`](https://zdebruine.github.io/RcppML/reference/st_write.md),
[`st_read`](https://zdebruine.github.io/RcppML/reference/st_read.md),
[`st_info`](https://zdebruine.github.io/RcppML/reference/st_info.md),
[`st_write_dense`](https://zdebruine.github.io/RcppML/reference/st_write_dense.md),
[`st_read_dense`](https://zdebruine.github.io/RcppML/reference/st_read_dense.md)
