# Run once before the test files.
#
# CRAN allows at most two cores. Every RcppML entry point resolves its thread
# count from the `threads` argument, then this option, then "all cores".
options(RcppML.threads = 2)
