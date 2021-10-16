#' Acute Myelogenous Leukemia cells
#'
#' @description ~800 differentially methylated regions (DMRs) in 123 Acute Myelogenous Leukemia (AML) samples, and 5 samples for putative cells of origin (GMP, LMPP, MEP) giving healthy DMR signatures.
#'
#' @details
#' AML methylation signatures differ from their cell of origin by additive methylation and subtractive methylation, in addition to tumor-specific heterogeneity. These AML tumors originated from one of three healthy cell types (GMP, LMPP, or MEP), the challenge is to classify the cell of origin based on these healthy cell type DMRs.
#'
#' @md
#' @docType data
#' @usage data(aml)
#' @format dense matrix of features (DMRs) as rows and samples ("AML sample", "GMP", "LMPP", or "MEP") as columns.
"aml"
