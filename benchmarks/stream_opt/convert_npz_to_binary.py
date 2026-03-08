#!/usr/bin/env python3
"""
Convert CellCensus_3M .npz chunks (CSR, cells×genes) to concatenated binary
arrays in CSC format (genes×cells) suitable for constructing R's dgCMatrix.

Usage: python3 convert_npz_to_binary.py <n_chunks> <output_dir>

Output files in <output_dir>:
  data.bin     float64 LE, nnz values
  indices.bin  int32 LE, row indices (0-based)
  indptr.bin   int32 LE, column pointers (ncol+1 entries)
  shape.txt    "nrow ncol nnz" on single line
"""

import sys
import os
import numpy as np
import scipy.sparse as sp
import time

DATA_DIR = "/mnt/projects/debruinz_project/CellCensus_3M"

def main():
    n_chunks = int(sys.argv[1])
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting {n_chunks} human chunks to binary CSC format...")

    # --- Pass 1: count total cells and nnz ---
    nrows = None  # genes (features) — same across chunks
    total_cols = 0  # cells across all chunks
    total_nnz = 0

    t0 = time.time()
    for i in range(1, n_chunks + 1):
        fname = os.path.join(DATA_DIR, f"3m_human_chunk_{i}.npz")
        npz = np.load(fname, allow_pickle=True)
        shape = tuple(npz["shape"])  # (cells, genes) in CSR
        nnz = len(npz["data"])
        if nrows is None:
            nrows = shape[1]  # genes dimension
        else:
            assert nrows == shape[1], f"Gene dimension mismatch: {nrows} vs {shape[1]}"
        total_cols += shape[0]  # cells become columns after transpose
        total_nnz += nnz
        print(f"  chunk {i}: {shape[0]} cells, {nnz:,} nnz")
        del npz

    print(f"\nTotal: {nrows} genes x {total_cols} cells, {total_nnz:,} nnz")
    print(f"Pass 1 took {time.time()-t0:.1f}s")

    # --- Check int32 overflow ---
    INT32_MAX = np.iinfo(np.int32).max
    if total_nnz > INT32_MAX:
        print(f"\nERROR: total nnz ({total_nnz:,}) exceeds int32 max ({INT32_MAX:,}).")
        print("R's dgCMatrix cannot handle this. Use fewer chunks.")
        sys.exit(1)

    # --- Allocate output arrays ---
    print(f"\nAllocating arrays: data={total_nnz*8/1e9:.1f}GB, "
          f"indices={total_nnz*4/1e9:.1f}GB, "
          f"indptr={(total_cols+1)*4/1e6:.1f}MB")

    data_out = np.empty(total_nnz, dtype=np.float64)
    indices_out = np.empty(total_nnz, dtype=np.int32)
    indptr_out = np.empty(total_cols + 1, dtype=np.int32)
    indptr_out[0] = 0

    # --- Pass 2: load chunks, convert to CSC, fill arrays ---
    t1 = time.time()
    col_offset = 0
    nnz_offset = 0

    for i in range(1, n_chunks + 1):
        tc = time.time()
        fname = os.path.join(DATA_DIR, f"3m_human_chunk_{i}.npz")
        npz = np.load(fname, allow_pickle=True)
        shape = tuple(npz["shape"])

        # Build CSR matrix (cells × genes)
        csr = sp.csr_matrix(
            (npz["data"], npz["indices"], npz["indptr"]),
            shape=shape
        )
        del npz

        # Transpose to genes × cells, then to CSC
        # csr.T gives CSC view; .tocsc() materializes as CSC
        csc = csr.T.tocsc()
        del csr

        chunk_ncols = csc.shape[1]
        chunk_nnz = csc.nnz

        # Copy data into output arrays
        data_out[nnz_offset:nnz_offset + chunk_nnz] = csc.data.astype(np.float64)
        indices_out[nnz_offset:nnz_offset + chunk_nnz] = csc.indices.astype(np.int32)

        # Adjust indptr: use int64 arithmetic to avoid overflow, then cast
        shifted = csc.indptr[1:].astype(np.int64) + np.int64(nnz_offset)
        indptr_out[col_offset + 1:col_offset + chunk_ncols + 1] = shifted.astype(np.int32)

        nnz_offset += chunk_nnz
        col_offset += chunk_ncols

        elapsed = time.time() - tc
        total_elapsed = time.time() - t1
        print(f"  chunk {i}/{n_chunks}: {chunk_ncols} cols, {chunk_nnz:,} nnz, "
              f"{elapsed:.1f}s (total {total_elapsed:.0f}s)")

        del csc

    assert nnz_offset == total_nnz
    assert col_offset == total_cols

    # --- Save binary files ---
    print(f"\nPass 2 took {time.time()-t1:.1f}s")
    print("Writing binary files...")

    ts = time.time()
    data_out.tofile(os.path.join(out_dir, "data.bin"))
    print(f"  data.bin: {total_nnz*8/1e9:.2f}GB ({time.time()-ts:.1f}s)")

    ts = time.time()
    indices_out.tofile(os.path.join(out_dir, "indices.bin"))
    print(f"  indices.bin: {total_nnz*4/1e9:.2f}GB ({time.time()-ts:.1f}s)")

    ts = time.time()
    indptr_out.tofile(os.path.join(out_dir, "indptr.bin"))
    print(f"  indptr.bin: {(total_cols+1)*4/1e6:.1f}MB ({time.time()-ts:.1f}s)")

    with open(os.path.join(out_dir, "shape.txt"), "w") as f:
        f.write(f"{nrows} {total_cols} {total_nnz}\n")
    print(f"  shape.txt: {nrows} x {total_cols}, {total_nnz:,} nnz")

    print(f"\nTotal conversion time: {time.time()-t0:.1f}s")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
