\name{kriging_weights}
\alias{kriging_weights}
\title{Extract Ordinary-Kriging Weights}
\usage{
kriging_weights(object, newdata)
}
\arguments{
  \item{object}{A fitted object returned by \code{ordinary_kriging()}.}
  \item{newdata}{Numeric matrix or data frame of prediction coordinates.}
}
\value{
A numeric matrix with prediction locations in rows and observations in columns.
}
\description{
Computes and transfers the unit-sum Ordinary-Kriging weights.
}
