\name{predict.geopytorch_ordinary_kriging}
\alias{predict.geopytorch_ordinary_kriging}
\title{Predict with a GeoPyTorch Ordinary-Kriging Model}
\usage{
\method{predict}{geopytorch_ordinary_kriging}(
  object,
  newdata,
  observation_noise = FALSE,
  full_covariance = FALSE,
  ...
)
}
\arguments{
  \item{object}{A fitted object returned by \code{ordinary_kriging()}.}
  \item{newdata}{Numeric matrix or data frame of prediction coordinates.}
  \item{observation_noise}{Include the fitted nugget in prediction variance.}
  \item{full_covariance}{Transfer the full joint covariance matrix to R.}
  \item{...}{Unused.}
}
\value{
An object of class \code{geopytorch_prediction} containing mean, marginal
variance, standard deviation, and optionally the joint covariance matrix.
}
\description{
Performs exact Ordinary-Kriging prediction on the device used for fitting and
materializes requested results as R numeric objects.
}
