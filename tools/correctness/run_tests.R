args <- commandArgs(TRUE)
filt <- if (nzchar(args[1])) args[1] else NULL
library(testthat); library(RcppML)
res <- as.data.frame(test_dir(".", filter = filt, reporter = "check", stop_on_failure = FALSE,
                              package = "RcppML", load_package = "installed"))
res$error <- as.integer(res$error)
agg <- aggregate(cbind(nb, failed, skipped, error, warning, real) ~ file, res, sum)
print(agg[order(-agg$failed, -agg$error, -agg$real), ], row.names = FALSE)
cat(sprintf("TOTAL files=%d tests=%d expectations=%d failed=%d errors=%d skipped=%d warnings=%d time=%.0fs\n",
            length(unique(res$file)), nrow(res), sum(res$nb), sum(res$failed), sum(res$error),
            sum(res$skipped), sum(res$warning), sum(res$real)))
cat("--- failing or erroring tests ---\n")
print(res[res$failed > 0 | res$error > 0, c("file", "test")], row.names = FALSE)
write.csv(res[, c("file", "test", "nb", "failed", "skipped", "error", "warning", "real")],
          file.path(Sys.getenv("RW"), "out", args[2]), row.names = FALSE)
