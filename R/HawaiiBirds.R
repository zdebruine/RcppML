#' Bird species frequency in Hawaii
#' 
#' @description Mean counts of bird species in 1-by-1km grids in Hawaii as recorded in the eBird project. Not all counts are complete because some grids are sampled better than others (i.e. urban areas, birding hotspots).
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
#'  * Mean counts of species in each grid across all checklists were compiled in the \code{$counts} component
#'  * Metadata for each grid was recorded in the \code{$metadata} component
#'
#' @md
#' @export
#' @docType data
#' @usage data(HawaiiBirds)
#' @format list of two components: \code{counts} giving mean total counts of species in each grid, and \code{metadata} giving information about each grid (i.e. latitude, longitude, and island)
#'
"HawaiiBirds"