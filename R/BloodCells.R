#' Interferon-stimulated blood cells
#' 
#' @description Transcript counts for 2000 variable genes in 1000 peripheral blood mononuclear cells (PBMCs), roughly half of them stimulated with IFNB
#' 
#' @details
#' This dataset was derived from https://www.nature.com/articles/nbt.4042 and SeuratData.
#'
#' @md
#' @export
#' @docType data
#' @usage data(BloodCells)
#' @format sparse \code{Matrix::dgCMatrix} of gene transcript counts (rows) vs. single cells (columns), labeled as either stimulated or not stimulated (colnames).
"BloodCells"