# Package index

## Non-negative Matrix Factorization

Core NMF functions and model class

- [`nmf()`](https://zdebruine.github.io/RcppML/reference/nmf.md) :
  Non-negative matrix factorization
- [`nmf-class`](https://zdebruine.github.io/RcppML/reference/nmf-class.md)
  : nmf S4 Class
- [`subset(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`` `[`( ``*`<nmf>`*`,`*`<ANY>`*`,`*`<ANY>`*`,`*`<ANY>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`head(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`show(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`dimnames(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`dim(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`t(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`sort(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`prod(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`` `$`( ``*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`coerce(`*`<nmf>`*`,`*`<list>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`` `[[`( ``*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  [`predict(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/nmf-class-methods.md)
  : nmf class methods
- [`summary(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/summary-nmf-method.md)
  [`plot(`*`<nmfSummary>`*`)`](https://zdebruine.github.io/RcppML/reference/summary-nmf-method.md)
  : Summarize NMF factors
- [`biplot(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/biplot-nmf-method.md)
  : Biplot for NMF factors
- [`nnls()`](https://zdebruine.github.io/RcppML/reference/nnls.md) :
  Non-negative Least Squares Projection
- [`compute_target()`](https://zdebruine.github.io/RcppML/reference/compute_target.md)
  : Compute a Target Matrix for Guided NMF
- [`refine()`](https://zdebruine.github.io/RcppML/reference/refine.md) :
  Refine an NMF Model Using Label-Guided Correction

## Singular Value Decomposition

SVD and PCA functions

- [`svd()`](https://zdebruine.github.io/RcppML/reference/svd.md) :
  Truncated SVD / PCA with constraints and regularization
- [`` `[`( ``*`<svd>`*`,`*`<ANY>`*`,`*`<ANY>`*`,`*`<ANY>`*`)`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  [`head(`*`<svd>`*`)`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  [`show(`*`<svd>`*`)`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  [`dim(`*`<svd>`*`)`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  [`reconstruct()`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  [`predict(`*`<svd>`*`)`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  [`variance_explained()`](https://zdebruine.github.io/RcppML/reference/svd-class.md)
  : svd S4 Class
- [`pca()`](https://zdebruine.github.io/RcppML/reference/pca.md) : PCA
  (centered SVD)

## Model Evaluation & Diagnostics

Functions for evaluating and comparing NMF models

- [`assess()`](https://zdebruine.github.io/RcppML/reference/assess.md) :
  Assess Embedding Quality
- [`as.data.frame(`*`<nmf_assessment>`*`)`](https://zdebruine.github.io/RcppML/reference/as.data.frame.nmf_assessment.md)
  : Convert assessment results to a one-row data frame
- [`print(`*`<nmf_assessment>`*`)`](https://zdebruine.github.io/RcppML/reference/print.nmf_assessment.md)
  : Print method for nmf_assessment objects
- [`evaluate()`](https://zdebruine.github.io/RcppML/reference/evaluate.md)
  : Evaluate an NMF model
- [`sparsity()`](https://zdebruine.github.io/RcppML/reference/sparsity.md)
  : Compute the sparsity of each NMF factor
- [`align()`](https://zdebruine.github.io/RcppML/reference/align.md) :
  Align two NMF models
- [`compare_nmf()`](https://zdebruine.github.io/RcppML/reference/compare_nmf.md)
  : Compare Multiple NMF Models
- [`cosine()`](https://zdebruine.github.io/RcppML/reference/cosine.md) :
  Cosine similarity

## Distribution Selection

Automatic distribution and zero-inflation diagnostics

- [`auto_nmf_distribution()`](https://zdebruine.github.io/RcppML/reference/auto_nmf_distribution.md)
  : Auto-select NMF distribution
- [`score_test_distribution()`](https://zdebruine.github.io/RcppML/reference/score_test_distribution.md)
  : Score-test distribution diagnostic
- [`diagnose_zero_inflation()`](https://zdebruine.github.io/RcppML/reference/diagnose_zero_inflation.md)
  : Diagnose zero inflation
- [`diagnose_dispersion()`](https://zdebruine.github.io/RcppML/reference/diagnose_dispersion.md)
  : Diagnose dispersion mode

## Visualization

Plotting functions for NMF results

- [`plot(`*`<nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/plot.nmf.md)
  : Plot NMF Training History and Diagnostics
- [`plot(`*`<nmfCrossValidate>`*`)`](https://zdebruine.github.io/RcppML/reference/plot.nmfCrossValidate.md)
  : Plot Cross-Validation Results

## Consensus Clustering

Multi-replicate consensus NMF

- [`consensus_nmf()`](https://zdebruine.github.io/RcppML/reference/consensus_nmf.md)
  : Consensus Clustering for NMF
- [`plot(`*`<consensus_nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/plot.consensus_nmf.md)
  : Plot Consensus Matrix Heatmap
- [`summary(`*`<consensus_nmf>`*`)`](https://zdebruine.github.io/RcppML/reference/summary.consensus_nmf.md)
  : Summary for Consensus NMF

## Clustering

NMF-based clustering algorithms

- [`bipartition()`](https://zdebruine.github.io/RcppML/reference/bipartition.md)
  : Bipartition a sample set
- [`dclust()`](https://zdebruine.github.io/RcppML/reference/dclust.md) :
  Divisive clustering
- [`plot(`*`<dclust>`*`)`](https://zdebruine.github.io/RcppML/reference/plot.dclust.md)
  : Plot divisive clustering hierarchy
- [`bipartiteMatch()`](https://zdebruine.github.io/RcppML/reference/bipartiteMatch.md)
  : Bipartite graph matching

## Composable Factorization Graphs

Build complex factorization pipelines with factor_net

- [`factor_net()`](https://zdebruine.github.io/RcppML/reference/factor_net.md)
  : Compile a factorization network
- [`factor_input()`](https://zdebruine.github.io/RcppML/reference/factor_input.md)
  : Create an input node for a factorization network
- [`factor_config()`](https://zdebruine.github.io/RcppML/reference/factor_config.md)
  : Global configuration for a factorization network
- [`nmf_layer()`](https://zdebruine.github.io/RcppML/reference/nmf_layer.md)
  : Create an NMF factorization layer
- [`svd_layer()`](https://zdebruine.github.io/RcppML/reference/svd_layer.md)
  : Create an SVD/PCA factorization layer
- [`factor_shared()`](https://zdebruine.github.io/RcppML/reference/factor_shared.md)
  : Shared factorization across multiple inputs (multi-modal)
- [`factor_concat()`](https://zdebruine.github.io/RcppML/reference/factor_concat.md)
  : Concatenate H factors from branches (row-bind)
- [`factor_add()`](https://zdebruine.github.io/RcppML/reference/factor_add.md)
  : Element-wise H addition (skip/residual connection)
- [`factor_condition()`](https://zdebruine.github.io/RcppML/reference/factor_condition.md)
  : Concatenate conditioning metadata to a layer's H
- [`fit()`](https://zdebruine.github.io/RcppML/reference/fit.md) : Fit a
  factorization network
- [`W()`](https://zdebruine.github.io/RcppML/reference/W.md)
  [`H()`](https://zdebruine.github.io/RcppML/reference/W.md) :
  Per-factor configuration for factorization layers
- [`cross_validate_graph()`](https://zdebruine.github.io/RcppML/reference/cross_validate_graph.md)
  : Cross-validate a factorization network
- [`predict(`*`<factor_net_result>`*`)`](https://zdebruine.github.io/RcppML/reference/predict.factor_net_result.md)
  : Project new data through a trained factor network
- [`training_logger()`](https://zdebruine.github.io/RcppML/reference/training_logger.md)
  : Create a training logger for factor network fitting
- [`export_log()`](https://zdebruine.github.io/RcppML/reference/export_log.md)
  : Export training log to CSV
- [`plot(`*`<training_logger>`*`)`](https://zdebruine.github.io/RcppML/reference/plot.training_logger.md)
  : Plot training log
- [`as.data.frame(`*`<training_logger>`*`)`](https://zdebruine.github.io/RcppML/reference/as.data.frame.training_logger.md)
  : Convert training log to data.frame
- [`classify_embedding()`](https://zdebruine.github.io/RcppML/reference/classify_embedding.md)
  : Evaluate classification performance of factor embeddings
- [`classify_logistic()`](https://zdebruine.github.io/RcppML/reference/classify_logistic.md)
  : Logistic regression classifier for factor embeddings
- [`classify_rf()`](https://zdebruine.github.io/RcppML/reference/classify_rf.md)
  : Random forest classifier for factor embeddings
- [`` `$`( ``*`<factor_net_result>`*`)`](https://zdebruine.github.io/RcppML/reference/cash-.factor_net_result.md)
  : Access layer results by name
- [`print(`*`<factor_net>`*`)`](https://zdebruine.github.io/RcppML/reference/print.factor_net.md)
  : Print a factor_net
- [`print(`*`<factor_net_cv>`*`)`](https://zdebruine.github.io/RcppML/reference/print.factor_net_cv.md)
  : Print a factor_net_cv result
- [`print(`*`<factor_net_result>`*`)`](https://zdebruine.github.io/RcppML/reference/print.factor_net_result.md)
  : Print a factor_net_result
- [`print(`*`<fn_factor_config>`*`)`](https://zdebruine.github.io/RcppML/reference/print.fn_factor_config.md)
  : Print an fn_factor_config
- [`print(`*`<fn_global_config>`*`)`](https://zdebruine.github.io/RcppML/reference/print.fn_global_config.md)
  : Print an fn_global_config
- [`print(`*`<fn_node>`*`)`](https://zdebruine.github.io/RcppML/reference/print.fn_node.md)
  : Print an fn_node
- [`print(`*`<fn_classifier_eval>`*`)`](https://zdebruine.github.io/RcppML/reference/print.fn_classifier_eval.md)
  : Print a classifier evaluation result
- [`summary(`*`<factor_net_result>`*`)`](https://zdebruine.github.io/RcppML/reference/summary.factor_net_result.md)
  : Summarize a factor_net_result
- [`summary(`*`<fn_classifier_eval>`*`)`](https://zdebruine.github.io/RcppML/reference/summary.fn_classifier_eval.md)
  : Summarize a classifier evaluation result
- [`print(`*`<training_logger>`*`)`](https://zdebruine.github.io/RcppML/reference/print.training_logger.md)
  : Print a training log

## Simulation

Functions for generating synthetic data

- [`simulateNMF()`](https://zdebruine.github.io/RcppML/reference/simulateNMF.md)
  : Simulate an NMF dataset
- [`simulateSwimmer()`](https://zdebruine.github.io/RcppML/reference/simulateSwimmer.md)
  : Simulate Swimmer Dataset

## StreamPress I/O

Streaming sparse matrix format (.spz)

- [`st_add_transpose()`](https://zdebruine.github.io/RcppML/reference/st_add_transpose.md)
  : Add Transpose Section to an Existing StreamPress File
- [`st_chunk_ranges()`](https://zdebruine.github.io/RcppML/reference/st_chunk_ranges.md)
  : Get Column Ranges for Each Chunk in a StreamPress File
- [`st_filter_cols()`](https://zdebruine.github.io/RcppML/reference/st_filter_cols.md)
  : Slice Columns Matching Variable Metadata Filter
- [`st_filter_rows()`](https://zdebruine.github.io/RcppML/reference/st_filter_rows.md)
  : Slice Rows Matching Observation Metadata Filter
- [`st_free_gpu()`](https://zdebruine.github.io/RcppML/reference/st_free_gpu.md)
  : Free GPU-Resident Sparse Matrix
- [`st_info()`](https://zdebruine.github.io/RcppML/reference/st_info.md)
  : Get metadata from a StreamPress file
- [`st_map_chunks()`](https://zdebruine.github.io/RcppML/reference/st_map_chunks.md)
  : Apply a Function to Every Chunk in a StreamPress File
- [`st_obs_indices()`](https://zdebruine.github.io/RcppML/reference/st_obs_indices.md)
  : Get Row Indices Matching Observation Metadata Filter
- [`st_read()`](https://zdebruine.github.io/RcppML/reference/st_read.md)
  : Read a StreamPress file into a dgCMatrix
- [`st_read_dense()`](https://zdebruine.github.io/RcppML/reference/st_read_dense.md)
  : Read a Dense Matrix from StreamPress v3 Format
- [`st_read_gpu()`](https://zdebruine.github.io/RcppML/reference/st_read_gpu.md)
  : Read StreamPress File Directly to GPU Memory
- [`st_read_obs()`](https://zdebruine.github.io/RcppML/reference/st_read_obs.md)
  : Read Observation (Row) Metadata from a StreamPress File
- [`st_read_var()`](https://zdebruine.github.io/RcppML/reference/st_read_var.md)
  : Read Variable (Column) Metadata from a StreamPress File
- [`st_slice()`](https://zdebruine.github.io/RcppML/reference/st_slice.md)
  : Slice Rows and/or Columns from a StreamPress File
- [`st_slice_cols()`](https://zdebruine.github.io/RcppML/reference/st_slice_cols.md)
  : Slice Columns from a StreamPress File
- [`st_slice_rows()`](https://zdebruine.github.io/RcppML/reference/st_slice_rows.md)
  : Slice Rows from a StreamPress File
- [`st_write()`](https://zdebruine.github.io/RcppML/reference/st_write.md)
  : Write a sparse matrix to a StreamPress file
- [`st_write_dense()`](https://zdebruine.github.io/RcppML/reference/st_write_dense.md)
  : Write a Dense Matrix to StreamPress v3 Format
- [`st_write_list()`](https://zdebruine.github.io/RcppML/reference/st_write_list.md)
  : Write a List of Matrices as a Single StreamPress File
- [`streampress`](https://zdebruine.github.io/RcppML/reference/streampress.md)
  : StreamPress I/O: Read, Write, and Inspect Compressed Matrices

## GPU Support

GPU acceleration functions

- [`gpu_available()`](https://zdebruine.github.io/RcppML/reference/gpu_available.md)
  : Check if GPU acceleration is available
- [`gpu_info()`](https://zdebruine.github.io/RcppML/reference/gpu_info.md)
  : Get GPU device information
- [`gpu-backend`](https://zdebruine.github.io/RcppML/reference/gpu-backend.md)
  : GPU NMF Backend
- [`print(`*`<gpu_sparse_matrix>`*`)`](https://zdebruine.github.io/RcppML/reference/gpu_sparse_matrix-methods.md)
  [`dim(`*`<gpu_sparse_matrix>`*`)`](https://zdebruine.github.io/RcppML/reference/gpu_sparse_matrix-methods.md)
  [`nrow(`*`<gpu_sparse_matrix>`*`)`](https://zdebruine.github.io/RcppML/reference/gpu_sparse_matrix-methods.md)
  [`ncol(`*`<gpu_sparse_matrix>`*`)`](https://zdebruine.github.io/RcppML/reference/gpu_sparse_matrix-methods.md)
  : Methods for gpu_sparse_matrix objects

## Datasets

Built-in datasets for examples and benchmarks

- [`aml`](https://zdebruine.github.io/RcppML/reference/aml.md) : Acute
  Myelogenous Leukemia (AML) Dataset
- [`digits`](https://zdebruine.github.io/RcppML/reference/digits.md) :
  MNIST Digits Dataset
- [`golub`](https://zdebruine.github.io/RcppML/reference/golub.md) :
  Golub ALL-AML Dataset (Brunet et al. 2004)
- [`hawaiibirds`](https://zdebruine.github.io/RcppML/reference/hawaiibirds.md)
  : Hawaii Bird Species Frequency Dataset
- [`movielens`](https://zdebruine.github.io/RcppML/reference/movielens.md)
  : MovieLens Dataset
- [`olivetti`](https://zdebruine.github.io/RcppML/reference/olivetti.md)
  : Olivetti Faces Dataset
- [`pbmc3k`](https://zdebruine.github.io/RcppML/reference/pbmc3k.md) :
  PBMC 3k Single-Cell RNA-seq Dataset (StreamPress Compressed)

## Package

- [`RcppML`](https://zdebruine.github.io/RcppML/reference/RcppML-package.md)
  [`RcppML-package`](https://zdebruine.github.io/RcppML/reference/RcppML-package.md)
  : RcppML: Fast Non-Negative Matrix Factorization and Divisive
  Clustering
