#' Acute Myelogenous Leukemia cells
#'
#' @description DNA methylation in ~800 regions in 123 Acute Myelogenous Leukemia (AML) samples classified by probable cell of origin (GMP, L-MPP, or MEP), and 
#' 5 samples from healthy references for each suspected cell of origin (GMP, L-MPP, MEP).
#'
#' @details
#' AML methylation signatures differ from their cell of origin by additive methylation and subtractive methylation, 
#' in addition to tumor-specific heterogeneity. These AML tumors likely originated from one of three healthy cell types (GMP, LMPP, or MEP), 
#' the challenge is to classify the cell of origin based on these healthy cell type DMRs.
#'
#' @md
#' @docType data
#' @usage data(aml)
#' @format list of dense matrix of features (methylated regions) as rows and samples ("AML" or "Control" samples, classified by putative cell of origin or reference cell type) as columns. The "metadata_h" maps to publicly available clinical metadata.
"aml"