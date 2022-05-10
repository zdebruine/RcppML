.onAttach <- function(libname, pkgname) {

  # ensure that RNG seed is initialized 
  if(!exists(".Random.seed")) set.seed(NULL)

  # set threads
  if(is.null(getOption("RcppML.threads"))) 
    options(RcppML.threads = 0)
  
  if(is.null(getOption("RcppML.verbose")))
    options(RcppML.verbose = FALSE)

  if(is.null(getOption("RcppML.debug")))
    options(RcppML.debug = TRUE)
  
  msg <- "RcppML v0.5.4 using 'options(RcppML.threads = "
  threads <- getOption("RcppML.threads")
  if(threads == 0) {
    msg <- c(msg, "0)' (all available threads)")
  } else msg <- c(msg, threads, ")'")
  msg <- c(msg, ", 'options(RcppML.verbose = ")
  if(getOption("RcppML.verbose")) {
    msg <- c(msg, "TRUE)'\n")
  } else msg <- c(msg, "FALSE)'")
#  msg <- c(msg, " and 'options(RcppML.debug = ")
#  if(getOption("RcppML.debug")) {
#    msg <- c(msg, "TRUE)'\n")
#  } else msg <- c(msg, "FALSE)'")
  packageStartupMessage(msg)
  invisible()
}
