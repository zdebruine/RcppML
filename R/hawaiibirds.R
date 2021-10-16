#' Bird species frequency in Hawaii
#'
#' @description Frequency of bird species observation within 1km-squared grids in Hawaii as recorded in the eBird project. Not all counts are complete because some grids are sampled better than others (i.e. urban areas, birding hotspots).
#'
#' @details
#' This dataset was obtained as follows:
#'
#'  * All eBird observations in Hawaii were downloaded from the eBird website as of Oct. 2021
#'  * Only complete checklists were retained
#'  * Only non-X counts were retained
#'  * Species were removed with fewer than 50 observations or that were spuhs
#'  * Grids were defined by latitude/longitude rounded to two decimal places
#'  * Grids were removed with fewer than 10 checklists
#'  * Mean frequency of species in each grid was calculated based on the number of times a species was observed and the number of checklists submitted in that grid.
#'  * Grids were assigned to one of each major Hawaiian island based on geographical coordinates.
#'
#' @md
#' @docType data
#' @usage data(hawaiibirds)
#' @format list of three components: \code{counts} giving mean total counts of species in each grid, \code{metadata_h} giving information about each grid (i.e. latitude, longitude, and island), and \code{metadata_w} giving information about species taxonomic classification.
#'
"hawaiibirds"
