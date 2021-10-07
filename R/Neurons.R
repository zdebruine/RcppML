#' Dopaminergic Neurons
#' 
#' @description Gene transcript counts from 243 Dopaminergic Neurons, 5000 highly variable genes expressed in at least 25 cells. Cell types as column names, genes as row names.
#' 
#' @details
#' This dataset was derived from La Manno (2016) and the scRNA-seq package (\code{LaMannoBrainData('mouse-adult')})
#'
#' @export
#' @references
#' La Manno, G., et al. 2016. "Molecular Diversity of Midbrain Development in Mouse, Human, and Stem Cells." Cell 167 (2): 566-80
#'
#' @docType data
#' @usage data(Neurons)
#' @format dense matrix of gene transcript counts (rows) vs. single cells (columns), labeled by cell type.
"Neurons"