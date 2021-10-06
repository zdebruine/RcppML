#' Bird species frequency in Hawaii
#' 
#' @description Mean counts of bird species in 10-by-10km grids in Hawaii.
#' 
#' @details 
#' This dataset was obtained as follows:
#' 
#'  * All eBird observations in Hawaii were downloaded from the eBird website as of Oct. 2021
#'  * Only complete checklists were retained
#'  * Only non-X counts were retained
#'  * Latitude/Longitude were rounded to one decimal place
#'  * Species were removed with fewer than 50 observations or that were spuhs
#'  * Grids were defined by latitude/longitude rounded to one decimal place
#'  * Grids were removed with fewer than 25 checklists
#'  * Mean counts of species in each grid across all checklists were compiled in the $counts component
#'  * Metadata for each grid was recorded in the $metadata component
#'
#' @md
#' @export
#' @docType data
#' @usage data(HawaiiBirds)
#' @format list of two components: \code{counts} giving mean total counts of species in each grid, and \code{metadata} giving information about each grid (i.e. latitude, longitude, and county)
#'
"HawaiiBirds"