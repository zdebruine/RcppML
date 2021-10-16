.onAttach <- function(libname, pkgname) {

  # set threads
  if(is.null(getOption("RcppML.threads"))) 
    options(RcppML.threads = 0)
  
  if(is.null(getOption("RcppML.verbose")))
    options(RcppML.verbose = FALSE)

  msg <- "RcppML v0.5.1 using 'options(RcppML.threads = "
  threads <- getOption("RcppML.threads")
  if(threads == 0) {
    msg <- c(msg, "0)' (all available threads)")
  } else msg <- c(msg, threads, ")'")
  msg <- c(msg, " and 'options(RcppML.verbose = ")
  if(getOption("RcppML.verbose")) {
    msg <- c(msg, "TRUE)'\n")
  } else msg <- c(msg, "FALSE)'")
  packageStartupMessage(msg)
  invisible()
}