# Avoid R CMD check notes for ggplot2 NSE variables
if(getRversion() >= "2.15.1")  utils::globalVariables(c(
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
