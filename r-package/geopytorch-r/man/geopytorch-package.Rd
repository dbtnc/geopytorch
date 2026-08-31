\name{geopytorch-package}
\alias{geopytorch}
\alias{geopytorch-package}
\title{R Interface to GeoPyTorch}
\description{
The \pkg{geopytorch} package exposes the GeoPyTorch Python package through
\pkg{reticulate}. Python is imported lazily. A set of helper functions is included for convenience.
}
\details{
Set the \env{GEOPYTORCH_PYTHON_REQUIREMENT} environment variable to a pip-style
local requirement such as
\code{geopytorch@/absolute/path/to/geopytorch}. When the variable is unset, the
package declares \code{geopytorch>=0.1.0}.
}
