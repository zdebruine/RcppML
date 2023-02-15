My list for implementing SVD based on NFM
=========================================

For SVD Class
----------
- Rename / refactor W and H functions
- Refactor fit function to perform SVD algorithm
- Add copy of nnls.hpp::c_nnls() that does not impose non-negativity


Possible Optimizations
----------------------
- Collapse loops for better multithreading in predict


Questions 
---------
- Best to make 'negative least squares' solver, or add negativity option to existing solver?
- Need to investigate the InnerIterator class, may need guidance