\name{kriging_weights}
\alias{kriging_weights}
\title{Extract Kriging Weights}
\usage{
kriging_weights(object, newdata, newcoords = NULL)
}
\arguments{
  \item{object}{A fitted object returned by \code{ordinary_kriging()} or \code{spatial_linear_model()}.}
  \item{newdata}{Prediction coordinates for ordinary kriging, or a prediction data frame for spatial linear models.}
  \item{newcoords}{Optional prediction coordinate matrix for spatial linear models.}
}
\value{
A numeric matrix with prediction locations in rows and observations in columns.
}
\description{
Computes ordinary- or universal-kriging weights. With a regression design matrix,
the weights reproduce the prediction design from the training design.
}
