import numpy as np
import scipy.sparse as sp
import os

datadir = "/mnt/projects/debruinz_project/CellCensus_3M"

total_cells = 0
total_nnz = 0

# Check all human chunks
for i in range(1, 31):
    fname = os.path.join(datadir, f"3m_human_chunk_{i}.npz")
    if not os.path.exists(fname):
        print(f"Chunk {i}: MISSING")
        continue
    fsize = os.path.getsize(fname) / (1024**3)
    data = np.load(fname, allow_pickle=True)
    shape = tuple(data["shape"])
    nnz = len(data["data"])
    dtype = data["data"].dtype
    density = nnz / (shape[0] * shape[1])
    total_cells += shape[0]
    total_nnz += nnz
    print(f"human_{i}: {shape[0]}x{shape[1]}, nnz={nnz:,}, density={density:.4f}, dtype={dtype}, file={fsize:.2f}GB")

print(f"\nHuman total: {total_cells:,} cells, {total_nnz:,} nnz")

# Check first mouse chunk
fname = os.path.join(datadir, "3m_mouse_chunk_1.npz")
data = np.load(fname, allow_pickle=True)
shape = tuple(data["shape"])
nnz = len(data["data"])
dt = data["data"].dtype
print(f"\nmouse_1: {shape[0]}x{shape[1]}, nnz={nnz:,}, dtype={dt}")

# Check metadata sample
import csv
meta = os.path.join(datadir, "3m_human_metadata_1.csv")
with open(meta) as f:
    reader = csv.reader(f)
    header = next(reader)
    print(f"\nMetadata columns: {header}")
    row1 = next(reader)
    print(f"First row: {row1[:5]}...")
