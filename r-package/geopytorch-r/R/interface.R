#' Selected GeoPyTorch Python requirement
#'
#' Reports the Python requirement selected when the package is loaded. Set
#' GEOPYTORCH_PYTHON_REQUIREMENT before loading the package to use a local
#' checkout, for example geopytorch@/absolute/path/to/geopytorch.
#'
#' @return A length-one character vector.
#' @export
geopytorch_requirement <- function() {
  .geopytorch_selected_requirement
}

#' Access the delayed GeoPyTorch Python module
#'
#' Returns the module proxy created during package loading. Merely retrieving
#' the proxy does not initialize Python; accessing one of its attributes does.
#'
#' @return A delayed Python module proxy.
#' @export
geopytorch_module <- function() {
  .geopytorch
}

.python_distribution_version <- function(metadata, distribution) {
  tryCatch(
    as.character(metadata$version(distribution)),
    error = function(error) NA_character_
  )
}

#' Report the active GeoPyTorch runtime
#'
#' Initializes Python, imports the selected GeoPyTorch module, and reports the
#' selected requirement, module source, and dependency versions.
#'
#' @return An object of class geopytorch_config.
#' @export
geopytorch_config <- function() {
  module <- geopytorch_module()
  module_path <- reticulate::py_to_r(module[["__file__"]])
  metadata <- reticulate::import("importlib.metadata", convert = TRUE)
  python <- reticulate::py_config()

  structure(
    list(
      r_package = as.character(utils::packageVersion("geopytorch")),
      requirement = geopytorch_requirement(),
      python_constraint = .geopytorch_selected_python_version,
      python = python$python,
      python_version = as.character(python$version),
      module_path = module_path,
      geopytorch = .python_distribution_version(metadata, "geopytorch"),
      torch = .python_distribution_version(metadata, "torch"),
      gpytorch = .python_distribution_version(metadata, "gpytorch"),
      numpy = .python_distribution_version(metadata, "numpy")
    ),
    class = "geopytorch_config"
  )
}

#' @export
print.geopytorch_config <- function(x, ...) {
  fields <- c(
    "R package" = x$r_package,
    "Python requirement" = x$requirement,
    "Python constraint" = x$python_constraint,
    "Python executable" = x$python,
    "Python version" = x$python_version,
    "GeoPyTorch module" = x$module_path,
    "GeoPyTorch version" = x$geopytorch,
    "Torch version" = x$torch,
    "GPyTorch version" = x$gpytorch,
    "NumPy version" = x$numpy
  )
  width <- max(nchar(names(fields)))
  for (index in seq_along(fields)) {
    cat(
      sprintf(
        "%-*s : %s\n",
        width,
        names(fields)[[index]],
        fields[[index]]
      )
    )
  }
  invisible(x)
}
