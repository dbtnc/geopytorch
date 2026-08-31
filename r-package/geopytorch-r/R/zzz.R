.geopytorch <- NULL
.geopytorch_selected_requirement <- NULL
.geopytorch_selected_python_version <- NULL

.geopytorch_python_version <- function() {
  system_name <- Sys.info()[["sysname"]]
  machine <- paste(Sys.info()[["machine"]], R.version$arch)
  intel_macos <- identical(system_name, "Darwin") &&
    grepl("x86_64|amd64", machine, ignore.case = TRUE)

  if (intel_macos) {
    ">=3.12,<3.13"
  } else {
    ">=3.12"
  }
}

.geopytorch_requirement_from_environment <- function() {
  override <- trimws(
    Sys.getenv("GEOPYTORCH_PYTHON_REQUIREMENT", unset = "")
  )
  if (nzchar(override)) {
    override
  } else {
    "geopytorch>=0.1.0"
  }
}

.onLoad <- function(libname, pkgname) {
  requirement <- .geopytorch_requirement_from_environment()
  python_version <- .geopytorch_python_version()
  reticulate::py_require(
    requirement,
    python_version = python_version
  )
  .geopytorch <<- reticulate::import(
    "geopytorch",
    delay_load = TRUE,
    convert = FALSE
  )
  .geopytorch_selected_requirement <<- requirement
  .geopytorch_selected_python_version <<- python_version
}
