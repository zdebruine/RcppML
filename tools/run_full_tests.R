#!/usr/bin/env Rscript
# Run full test suite and write results to file
res <- devtools::test("/mnt/home/debruinz/RcppML-2")
df <- as.data.frame(res)
summary_line <- sprintf("FAIL: %d | WARN: %d | SKIP: %d | PASS: %d",
                        sum(df$failed), sum(df$warning > 0),
                        sum(df$skipped > 0), sum(df$passed))
cat(summary_line, "\n")
writeLines(summary_line, "/mnt/home/debruinz/RcppML-2/test_summary.txt")

# Write any failures
if (sum(df$failed) > 0) {
  failed_tests <- df[df$failed > 0, c("file", "test", "failed")]
  write.csv(failed_tests, "/mnt/home/debruinz/RcppML-2/test_failures.csv", row.names = FALSE)
}
