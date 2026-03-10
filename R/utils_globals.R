# Avoid R CMD check notes for ggplot2 NSE variables and internal helpers
if(getRversion() >= "2.15.1")  utils::globalVariables(c(
  ".st_dispatch",
  "iteration",
  "train_loss",
  "test_loss",
  "model",
  "loss",
  "loss_type",
  "loss_smooth",
  "component",
  "factor",
  "stat",
  "group",
  "factor1",
  "factor2",
  "labels",
  "value"
))
