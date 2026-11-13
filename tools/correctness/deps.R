need <- c("Rcpp", "RcppEigen", "Matrix", "testthat", "knitr", "rmarkdown", "ggplot2", "jsonlite",
          "pheatmap", "randomForest", "RColorBrewer", "patchwork", "plotly", "scales", "uwot",
          "viridis", "hdf5r", "SeuratObject", "Seurat")
have <- function(p) vapply(p, requireNamespace, logical(1), quietly = TRUE)
miss <- need[!have(need)]
cat("missing:", miss, "\n")
if (length(miss)) install.packages(miss, lib = Sys.getenv("R_LIBS_USER"), repos = "https://cloud.r-project.org", Ncpus = 16)
cat("STILL MISSING:", need[!have(need)], "\n")
