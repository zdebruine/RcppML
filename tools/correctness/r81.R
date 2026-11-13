library(RcppML); data(aml)
for (tt in c(FALSE, TRUE)) for (cs in c(123, 125)) {
  cv <- RcppML::nmf(aml, k = 2:10, test_fraction = 0.05, track_train_loss = tt, seed = 12, cv_seed = cs)
  cat("\n--- track_train_loss =", tt, " cv_seed =", cs, " class:", class(cv), "\n")
  print(as.data.frame(cv))
}
