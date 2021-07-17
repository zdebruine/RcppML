#' 7500 cells from mouse embryos
#'
#' @description This dataset is ~95% sparse and is constituted of a large number of mixed signals with significant signal dropout and technical noise.
#'
#' @details
#' The dataset was obtained as follows:
#'
#'      1. Pre-processed counts were downloaded from the Mouse Organogenesis Cell Atlas figshare site ("gene_counts_cleaned.rds")
#'
#'      2. Cells from E13.5 embryos were subsetted
#'
#'      3. Protein-coding genes were subsetted (17560 features out of original 26183)
#'
#'      4. Cells were subsetted which had non-zero counts for >500 genes (108825 cells out of original 241800)
#'
#'      5. Genes were subsetted which had non-zero counts in at least 0.5% (544 cells) of the cells (11379 genes out of original 17560)
#'
#'      6. 7500 cells were randomly selected
#'
#' @docType data
#'
#' @usage data(moca7k)
#'
#' @format dgCMatrix with features (genes) as rows and samples (cells) as columns
#'
#' @references Cao et al. (2019) Nature
#' (\href{https://pubmed.ncbi.nlm.nih.gov/30787437/}{PubMed})
#'
#' @source \href{https://oncoscape.v3.sttrcancer.org/atlas.gs.washington.edu.mouse.rna/downloads}{MOCA figshare}
#'
"moca7k"
