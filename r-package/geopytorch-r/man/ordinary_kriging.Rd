\name{ordinary_kriging}
\alias{ordinary_kriging}
\title{Fit an Ordinary-Kriging Model with GeoPyTorch}
\usage{
ordinary_kriging(
  train_x,
  train_y,
  covariance_model = c("matern", "exponential", "gaussian", "spherical", "wave"),
  matern_nu = 1.5,
  ard = FALSE,
  partial_sill = NULL,
  phi = NULL,
  nugget = NULL,
  nugget_lower_bound = 0,
  training_iterations = 100L,
  learning_rate = 0.1,
  likelihood_method = c("ml", "reml"),
  optimizer = c("adam", "lbfgs"),
  device = "cpu",
  dtype = c("float64", "float32")
)
}
\arguments{
  \item{train_x}{Numeric matrix or data frame of observation coordinates.}
  \item{train_y}{Numeric vector of observed values.}
  \item{covariance_model}{Covariance model name.}
  \item{matern_nu}{Matérn smoothness: 0.5, 1.5, or 2.5.}
  \item{ard}{Whether to use a range per coordinate.}
  \item{partial_sill}{Optional fixed or initial covariance partial sill.}
  \item{phi}{Optional fixed or initial covariance range.}
  \item{nugget}{Optional fixed or initial Gaussian noise variance.}
  \item{nugget_lower_bound}{Strict lower bound used when a nugget is supplied.}
  \item{training_iterations}{Optimizer iterations; zero retains initialized parameters.}
  \item{learning_rate}{Positive optimizer learning rate.}
  \item{likelihood_method}{Maximum likelihood (\code{"ml"}) or restricted maximum likelihood (\code{"reml"}).}
  \item{optimizer}{\code{"adam"} or \code{"lbfgs"}.}
  \item{device}{PyTorch device. CPU is the default; use \code{"cuda"} for the first CUDA GPU.}
  \item{dtype}{Either \code{"float64"} or \code{"float32"}.}
}
\value{
An object of class \code{geopytorch_ordinary_kriging}.
}
\description{
Converts R coordinates and values to tensors on the requested device and fits
GeoPyTorch's exact Ordinary-Kriging model.
}
