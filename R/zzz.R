# Package load hook and small internal helpers shared by all entry points.

.onLoad <- function(libname, pkgname) {
  defaults <- list(
    RcppML.threads = 0L,     # 0 = all available cores
    RcppML.verbose = FALSE
  )
  unset <- !(names(defaults) %in% names(options()))
  if (any(unset)) options(defaults[unset])
  invisible()
}

# Package-level state (one-time messages)
.rcppml_state <- new.env(parent = emptyenv())

#' Resolve the number of OpenMP threads
#'
#' Single place where every entry point decides how many threads to use.
#' Order of precedence: a non-zero \code{threads} argument, then
#' \code{getOption("RcppML.threads")}, then all cores (\code{0}).
#' The result is capped at 2 when \code{_R_CHECK_LIMIT_CORES_} is set to a
#' true value, and at \code{OMP_THREAD_LIMIT} when that variable is set.
#'
#' @param threads requested thread count; \code{NULL}, \code{NA} or \code{0}
#'   mean "not specified".
#' @return A non-negative integer; \code{0} means all available cores.
#' @keywords internal
#' @noRd
.resolve_threads <- function(threads = 0L) {
  as_count <- function(x) {
    if (is.null(x) || length(x) != 1L) return(0L)
    x <- suppressWarnings(as.integer(x))
    if (is.na(x) || x < 0L) 0L else x
  }
  n <- as_count(threads)
  if (n == 0L) n <- as_count(getOption("RcppML.threads", 0L))

  cap <- 0L
  chk <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_", ""))
  if (nzchar(chk) && !(chk %in% c("false", "f", "no", "0"))) cap <- 2L
  omp <- as_count(Sys.getenv("OMP_THREAD_LIMIT", ""))
  if (omp > 0L) cap <- if (cap > 0L) min(cap, omp) else omp

  if (cap > 0L && (n == 0L || n > cap)) n <- cap
  n
}

# Snapshot / restore of the user's RNG state, used around internal
# set.seed() calls so that a numeric `seed` argument does not change the
# caller's random number stream.
.get_rng_state <- function() {
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  } else {
    NULL
  }
}

.set_rng_state <- function(state) {
  if (is.null(state)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
      rm(".Random.seed", envir = globalenv())
  } else {
    assign(".Random.seed", state, envir = globalenv())
  }
  invisible()
}

# Emit a message at most once per session, keyed by `id`.
.message_once <- function(id, ...) {
  if (!isTRUE(.rcppml_state[[id]])) {
    .rcppml_state[[id]] <- TRUE
    message(...)
  }
  invisible()
}
