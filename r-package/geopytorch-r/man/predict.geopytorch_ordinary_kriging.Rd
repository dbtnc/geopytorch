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
\examples{
\dontrun{
# Needs 'Python'; on first use 'reticulate' installs GeoPyTorch and PyTorch.
train_x <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
train_y <- c(1, 2, 4, 3)
target_x <- matrix(c(0.25, 0.4, 0.8, 0.6), ncol = 2, byrow = TRUE)
fit <- ordinary_kriging(
  train_x,
  train_y,
  covariance_model = "exponential",
  partial_sill = 1.2,
  phi = 0.7,
  nugget = 1e-3,
  training_iterations = 0
)
prediction <- predict(fit, target_x, observation_noise = TRUE)
prediction$mean
prediction$variance
}
}
