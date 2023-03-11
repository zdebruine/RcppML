setClass("svd",
  representation(u = "matrix", v = "matrix", d = "numeric", misc = "list"),
  prototype(u = matrix(), v = matrix(), d = NA_real_, misc = list()),
  validity = function(object) {
    msg <- NULL
    if (ncol(object@u) != ncol(object@v))
      msg <- c(msg, "ranks of 'u' and 'v' are not equal")
    if (is.null(msg)) TRUE else msg
  })


setMethod("$", signature = "svd", function(x, name) {
  validObject(x)
  name <- tolower(name)
  if (!(name %in% c("u", "v", "d", "misc"))) {
    if (!(name %in% names(x@misc))) {
      stop("'name' not a slot in 'object' or component in 'object@misc'")
    } else x@misc[[name]]
  } else slot(x, name)
})