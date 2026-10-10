test_that("Ordinary Kriging uses CPU by default and returns R predictions", {
  checkout <- Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT", unset = "")
  skip_if(!nzchar(checkout), "no live checkout was requested")

  train_x <- matrix(
    c(0, 0, 1, 0, 0, 1, 1, 1),
    ncol = 2,
    byrow = TRUE
  )
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

  expect_s3_class(fit, "geopytorch_ordinary_kriging")
  expect_identical(fit$device, "cpu")
  prediction <- predict(fit, target_x, full_covariance = TRUE)
  expect_s3_class(prediction, "geopytorch_prediction")
  expect_length(prediction$mean, 2L)
  expect_length(prediction$variance, 2L)
  expect_equal(dim(prediction$covariance_matrix), c(2L, 2L))
  expect_true(all(is.finite(prediction$mean)))
  expect_true(all(prediction$variance >= 0))

  weights <- kriging_weights(fit, target_x)
  expect_equal(dim(weights), c(2L, 4L))
  expect_equal(rowSums(weights), c(1, 1), tolerance = 1e-10)
})

test_that("Ordinary-Kriging predictions preserve constant shifts", {
  checkout <- Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT", unset = "")
  skip_if(!nzchar(checkout), "no live checkout was requested")

  train_x <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
  train_y <- c(1, 2, 4, 3)
  target_x <- matrix(c(0.25, 0.4, 0.8, 0.6), ncol = 2, byrow = TRUE)
  fit_model <- function(values) {
    ordinary_kriging(
      train_x,
      values,
      covariance_model = "exponential",
      partial_sill = 1.2,
      phi = 0.7,
      nugget = 1e-3,
      training_iterations = 0
    )
  }

  base <- predict(fit_model(train_y), target_x)
  shifted <- predict(fit_model(train_y + 7), target_x)
  expect_equal(shifted$mean, base$mean + 7, tolerance = 1e-10)
  expect_equal(shifted$variance, base$variance, tolerance = 1e-10)
})

test_that("device helpers retain CPU as the portable default", {
  checkout <- Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT", unset = "")
  skip_if(!nzchar(checkout), "no live checkout was requested")

  expect_true("cpu" %in% geopytorch_devices())
  expect_identical(geopytorch_device(), "cpu")
  expect_true(geopytorch_device("auto") %in% geopytorch_devices())
})

test_that("R-side input validation catches incompatible shapes", {
  checkout <- Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT", unset = "")
  skip_if(!nzchar(checkout), "no live checkout was requested")

  expect_error(
    ordinary_kriging(matrix(1:4, ncol = 2), 1),
    "same number of rows"
  )
  fit <- ordinary_kriging(
    matrix(c(0, 0, 1, 1), ncol = 2, byrow = TRUE),
    c(0, 1),
    training_iterations = 0
  )
  expect_error(predict(fit, matrix(1:3, nrow = 1)), "must have 2 columns")
})
