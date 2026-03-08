library(SeuratData)
ad <- AvailableData()
installed <- ad[ad$Installed,]
cat("=== INSTALLED DATASETS ===\n")
for (i in seq_len(nrow(installed))) {
  cat(sprintf("  %s: %s\n", rownames(installed)[i], installed$Summary[i]))
}
cat("\n=== NOT INSTALLED ===\n")
not_installed <- ad[!ad$Installed,]
for (i in seq_len(nrow(not_installed))) {
  cat(sprintf("  %s: %s\n", rownames(not_installed)[i], not_installed$Summary[i]))
}
