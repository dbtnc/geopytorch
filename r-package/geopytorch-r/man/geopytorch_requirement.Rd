\name{geopytorch_requirement}
\alias{geopytorch_requirement}
\title{Selected GeoPyTorch Python Requirement}
\usage{
geopytorch_requirement()
}
\value{
A length-one character vector.
}
\description{
Reports the Python requirement selected when the package is loaded.
}
\details{
Set \env{GEOPYTORCH_PYTHON_REQUIREMENT} before loading the package to use a
local checkout, for example
\code{geopytorch@/absolute/path/to/geopytorch}.
}
\examples{
geopytorch_requirement()
}
