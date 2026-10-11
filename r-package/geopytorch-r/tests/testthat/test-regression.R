regression_data <- function() {
  set.seed(12)
  data.frame(
    s1 = runif(20),
    s2 = runif(20),
    z = rnorm(20),
    group = factor(rep(c("a", "b"), 10)),
    y = rnorm(20)
  )
}

regression_fixture <- function(
  formula = y ~ z + group,
  coords = c("s1", "s2")
) {
  skip_if(
    !nzchar(Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT")),
    "no live checkout was requested"
  )
  splm(
    formula,
    regression_data(),
    coords,
    initial = c(1.2, .2, .1),
    training_iterations = 0L
  )
}

test_that("formula regression preserves design, factor levels and uncertainty", {
  fit <- regression_fixture()
  d <- regression_data()
  new <- d[1:4, ]
  new$s1 <- new$s1 + .015
  new$group <- factor(rep("b", 4)) # subset levels must use training contrasts
  latent <- predict(fit, new, full_covariance = TRUE)
  observed <- predict(
    fit,
    new,
    observation_noise = TRUE,
    full_covariance = TRUE
  )
  expect_s3_class(fit, "geopytorch_splm")
  expect_identical(names(coef(fit)), c("(Intercept)", "z", "groupb"))
  expect_equal(dim(vcov(fit)), c(3L, 3L))
  expect_equal(
    observed$covariance_matrix - latent$covariance_matrix,
    diag(.1, 4),
    tolerance = 1e-10
  )
  expect_equal(
    kriging_weights(fit, new) %*% fit$design_matrix,
    unname(model.matrix(
      ~ z + group,
      transform(new, group = factor(group, levels = c("a", "b")))
    )),
    ignore_attr = TRUE,
    tolerance = 1e-9
  )
  expect_equal(attr(logLik(fit), "df"), 0L)
  expect_equal(as.numeric(logLik(fit)), fit$restricted_log_likelihood)
  bad <- new
  bad$group <- "unknown"
  expect_error(predict(fit, bad), "new level")
  bad <- new
  bad$z[1] <- NA_real_
  expect_error(predict(fit, bad), "missing")
})

test_that("matrix coordinates and no-intercept formulas work", {
  fit <- regression_fixture(
    y ~ z - 1,
    as.matrix(regression_data()[c("s1", "s2")])
  )
  new <- regression_data()[1:3, ]
  expect_error(predict(fit, new), "supply newcoords")
  p <- predict(fit, new, newcoords = as.matrix(new[c("s1", "s2")]))
  expect_length(p$mean, 3L)
  expect_length(coef(fit), 1L)
})

test_that("polynomial prediction uses the training basis", {
  fit <- regression_fixture(y ~ poly(z, 2))
  new <- regression_data()[c(2, 7, 14), ]
  X <- model.matrix(delete.response(fit$terms), new)
  expect_equal(
    kriging_weights(fit, new) %*% fit$design_matrix,
    unname(X),
    ignore_attr = TRUE,
    tolerance = 1e-8
  )
})

test_that("regression rejects invalid training designs", {
  skip_if(
    !nzchar(Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT")),
    "no live checkout was requested"
  )
  d <- regression_data()
  d$duplicate <- d$z
  expect_error(
    splm(y ~ z + duplicate, d, c("s1", "s2")),
    "full column rank"
  )
  expect_error(
    splm(y ~ z + offset(z), d, c("s1", "s2")),
    "offsets"
  )
  expect_error(splm(y ~ z, d, "absent"), "coordinate columns")
  d$y[1] <- NA_real_
  expect_error(splm(y ~ z, d, c("s1", "s2")), "missing")
})

test_that("bounded LBFGS fits expose diagnostics through reticulate", {
  skip_if(
    !nzchar(Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT")),
    "no live checkout was requested"
  )
  d <- data.frame(s1 = seq_len(8), y = rep(c(-3, 3), 4))
  fit <- splm(
    y ~ 1,
    d,
    "s1",
    covariance_model = "spherical",
    likelihood_method = "ml",
    initial = c(.5, .25, .1),
    parameter_bounds = list(c(0, 1), c(0, .5), c(0, .2))
  )
  expected <- -.5 * (8 * log(2 * pi * 1.2) + sum(d$y^2) / 1.2)
  expect_true(fit$converged)
  expect_equal(as.numeric(logLik(fit)), expected, tolerance = 2e-6)
  expect_equal(
    unname(fit$covariance_parameters[c(1, 3)]),
    c(1, .2),
    tolerance = 2e-6
  )
  expect_identical(fit$optimizer_results[[1]]$optimizer, "torch.optim.LBFGS")
  expect_true(fit$optimizer_results[[1]]$at_bound)
})
