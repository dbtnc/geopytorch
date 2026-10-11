\name{kriging_weights}
\alias{kriging_weights}
\title{Extract Kriging Weights}
\usage{
kriging_weights(object, newdata, newcoords = NULL)
}
\arguments{
  \item{object}{A fitted object returned by \code{ordinary_kriging()}, \code{splm()}, or \code{spglm()}.}
  \item{newdata}{Prediction coordinates for ordinary kriging, or a prediction data frame for spatial linear models.}
  \item{newcoords}{Optional prediction coordinate matrix for spatial linear models.}
}
\value{
A numeric matrix with prediction locations in rows and observations in columns.
For \code{spglm()}, weights act on the offset-free fitted latent mode.
}
\description{
Computes ordinary- or universal-kriging weights. With a regression design matrix,
the weights reproduce the prediction design from the training design.
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
weights <- kriging_weights(fit, target_x)
rowSums(weights)
}
}
