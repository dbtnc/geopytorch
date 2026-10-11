\name{geopytorch_threads}
\alias{geopytorch_threads}
\title{Get or Set PyTorch CPU Thread Counts}
\usage{
geopytorch_threads(threads = NULL, interop_threads = NULL)
}
\arguments{
  \item{threads}{Optional positive intra-op thread count.}
  \item{interop_threads}{Optional positive inter-op thread count.}
}
\value{
A list containing effective intra-op and inter-op thread counts.
}
\description{
Initializes Python and optionally configures PyTorch CPU parallelism.
PyTorch permits changing the inter-op count only before parallel work starts.
}
\examples{
\dontrun{
# Needs 'Python'; on first use 'reticulate' installs GeoPyTorch and PyTorch.
geopytorch_threads()
geopytorch_threads(threads = 2)
}
}
