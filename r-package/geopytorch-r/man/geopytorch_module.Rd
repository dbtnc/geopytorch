\name{geopytorch_module}
\alias{geopytorch_module}
\title{Access the Delayed GeoPyTorch Python Module}
\usage{
geopytorch_module()
}
\value{
A delayed Python module proxy.
}
\description{
Returns the module proxy created during package loading. Retrieving the proxy
does not initialize Python; accessing one of its attributes does.
}
\examples{
# Getting the module does not start Python.
module <- geopytorch_module()

\dontrun{
# Using one of its attributes does.
module$COVARIANCE_MODELS
}
}
