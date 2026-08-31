\name{geopytorch_devices}
\alias{geopytorch_devices}
\alias{geopytorch_device}
\title{Inspect and Select PyTorch Devices}
\usage{
geopytorch_devices()

geopytorch_device(device = "cpu", dtype = c("float64", "float32"))
}
\arguments{
  \item{device}{A device string: \code{"cpu"}, \code{"auto"},
  \code{"mps"}, \code{"cuda"}, or \code{"cuda:N"}.}
  \item{dtype}{Either \code{"float64"} or \code{"float32"}.}
}
\value{
\code{geopytorch_devices()} returns available device strings.
\code{geopytorch_device()} returns one validated device string.
}
\description{
Initializes Python and inspects or validates devices in the selected PyTorch
runtime. CPU is always available. Automatic selection prefers CUDA, then MPS
for single precision, then CPU.
}
