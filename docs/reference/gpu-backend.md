# GPU NMF Backend

Internal functions for GPU-accelerated NMF via CUDA. Automatically
detects GPU availability and dispatches to GPU when beneficial, falling
back to CPU otherwise.
