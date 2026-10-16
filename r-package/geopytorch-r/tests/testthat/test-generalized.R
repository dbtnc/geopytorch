glm_fixture <- function(
  family = "poisson",
  formula = y ~ z + group + offset(log(exposure))
) {
  skip_if(
    !nzchar(Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT")),
    "no live checkout was requested"
  )
  set.seed(12)
  d <- data.frame(
    s1 = runif(20),
    s2 = runif(20),
    z = rnorm(20),
    group = factor(rep(c("a", "b"), 10))
  )
  d$exposure <- rep(c(.8, 1.2), 10)
  d$y <- rep(c(1, 3, 2, 5, 4), 4)
  if (family == "binomial") {
    d$failures <- 8 - d$y
    formula <- cbind(y, failures) ~ z + group + offset(log(exposure))
  }
  if (family == "tweedie") {
    d$y <- rep(c(0, .2, .8, 1.5, 3), 4)
  }
  if (family == "beta") {
    d$y <- d$y / 6
  }
  fit <- spglm(
    formula,
    family,
    d,
    c("s1", "s2"),
    initial = c(.7, .3, .2),
    dispersion = if (family %in% c("poisson", "binomial")) 1 else 3,
    training_iterations = 0L
  )
  list(fit = fit, data = d)
}

test_that("all families fit through reticulate and support R generics", {
  for (family in c(
    "poisson",
    "nbinomial",
    "binomial",
    "beta",
    "Gamma",
    "inverse.gaussian",
    "tweedie"
  )) {
    setup <- glm_fixture(family)
    fit <- setup$fit
    expect_s3_class(fit, "geopytorch_spglm")
    expect_true(fit$converged)
    expect_identical(names(coef(fit)), c("(Intercept)", "z", "groupb"))
    expect_equal(dim(vcov(fit)), c(3L, 3L))
    expect_equal(nobs(fit), 20L)
    expect_equal(attr(logLik(fit), "df"), 0L)
    previous <- fit
    previous$power_estimated <- NULL
    expect_equal(logLik(previous), logLik(fit))
    expect_equal(residuals(fit), fit$response - fitted(fit))
    expect_s3_class(summary(fit), "summary.geopytorch_spglm")
    expect_output(print(fit), "spatial GLM")
    expect_output(print(summary(fit)), "spatial GLM")
    new <- setup$data[1:3, ]
    new$s1 <- new$s1 + .025
    new$group <- factor(rep("b", 3))
    link <- predict(fit, new, type = "link", full_covariance = TRUE)
    response <- predict(fit, new, full_covariance = TRUE)
    observed <- predict(fit, new, observation_noise = TRUE)
    expect_length(link$mean, 3L)
    expect_equal(dim(link$covariance_matrix), c(3L, 3L))
    expect_true(all(observed$variance > response$variance))
    invlink <- if (family %in% c("binomial", "beta")) plogis else exp
    expect_equal(response$mean, invlink(link$mean), tolerance = 1e-9)
    expect_error(
      predict(fit, new, type = "link", observation_noise = TRUE),
      "response scale"
    )
    if (family == "binomial") {
      counts <- predict(fit, new, trials = c(2, 3, 4))
      expect_equal(counts$mean, response$mean * c(2, 3, 4), tolerance = 1e-9)
      expect_equal(
        counts$variance,
        response$variance * c(2, 3, 4)^2,
        tolerance = 1e-9
      )
    }
  }
})

test_that("formula offsets and training contrasts are retained", {
  setup <- glm_fixture()
  d <- setup$data[1:4, ]
  baseline <- predict(setup$fit, d, type = "link")
  d$exposure <- 2 * d$exposure
  changed <- predict(setup$fit, d, type = "link")
  expect_equal(changed$mean - baseline$mean, rep(log(2), 4), tolerance = 1e-10)
  expect_equal(changed$variance, baseline$variance, tolerance = 1e-10)
  d$group <- "unknown"
  expect_error(predict(setup$fit, d), "new level")
  d <- setup$data
  d$duplicate <- d$z
  expect_error(
    spglm(y ~ z + duplicate, "poisson", d, c("s1", "s2")),
    "full column rank"
  )
})

test_that("binomial factors and explicit totals agree with matrix responses", {
  setup <- glm_fixture("binomial")
  fit <- spglm(
    y ~ z + group + offset(log(exposure)),
    "binomial",
    setup$data,
    c("s1", "s2"),
    trials = rep(8, 20),
    initial = c(.7, .3, .2),
    training_iterations = 0L
  )
  expect_equal(coef(fit), coef(setup$fit), tolerance = 1e-10)
  d <- setup$data
  d$y <- factor(rep(c("no", "yes"), 10))
  binary <- spglm(
    y ~ z,
    "binomial",
    d,
    c("s1", "s2"),
    initial = c(.7, .3, .2),
    training_iterations = 0L
  )
  expect_equal(binary$response, rep(c(0, 1), 10))
  expect_error(
    spglm(y ~ z, "binomial", d, c("s1", "s2"), trials = 2),
    "no trials"
  )
})

test_that("ml uses marginal Laplace coefficients and reports their df", {
  setup <- glm_fixture()
  ml <- spglm(
    y ~ z + group + offset(log(exposure)),
    "poisson",
    setup$data,
    c("s1", "s2"),
    initial = c(.7, .3, .2),
    likelihood_method = "ml",
    training_iterations = 0L
  )
  expect_true(ml$converged)
  expect_equal(attr(logLik(ml), "df"), 3L)
  expect_equal(as.numeric(logLik(ml)), ml$log_likelihood)
  expect_true(all(is.finite(coef(ml))))
  expect_true(all(eigen(vcov(ml), only.values = TRUE)$values > 0))
  # REML coefficients are joint modes, so the two estimators differ.
  expect_false(isTRUE(all.equal(coef(ml), coef(setup$fit), tolerance = 1e-8)))
  expect_length(kriging_weights(setup$fit, setup$data[1:2, ]), 40L)
  expect_error(kriging_weights(ml, setup$data[1:2, ]), "REML")
})

test_that("responses without finite coefficients are rejected", {
  setup <- glm_fixture()
  d <- setup$data
  d$y <- as.numeric(d$group == "b")
  for (method in c("reml", "ml")) {
    expect_error(
      spglm(
        y ~ group,
        "binomial",
        d,
        c("s1", "s2"),
        likelihood_method = method,
        training_iterations = 0L
      ),
      "no finite coefficient"
    )
  }
  d$y <- 0
  expect_error(
    spglm(y ~ z, "poisson", d, c("s1", "s2"), training_iterations = 0L),
    "no finite coefficient"
  )
})

test_that("the Gaussian R entry point is splm", {
  expect_true(is.function(splm))
  expect_false("spatial_linear_model" %in% getNamespaceExports("geopytorch"))
})

test_that("Tweedie power controls and response variance cross the R interface", {
  setup <- glm_fixture("tweedie")
  fit <- setup$fit
  expect_equal(fit$power, 1.5)
  expect_false(fit$power_estimated)
  expect_equal(summary(fit)$power, 1.5)
  expect_output(print(summary(fit)), "Tweedie power")
  predicted <- predict(fit, setup$data)
  observed <- predict(fit, setup$data, observation_noise = TRUE)
  expect_equal(
    observed$variance - predicted$variance,
    fit$dispersion * predicted$mean^fit$power,
    tolerance = 1e-8
  )
  for (estimate in c(FALSE, TRUE)) {
    estimated <- spglm(
      y ~ z,
      "tweedie",
      setup$data,
      c("s1", "s2"),
      initial = c(.3, .3, .1),
      fixed = c(TRUE, TRUE, TRUE),
      power = 1.4,
      estimate_power = estimate,
      training_iterations = 150L
    )
    expect_true(estimated$converged)
    expect_identical(estimated$power_estimated, estimate)
    expect_equal(attr(logLik(estimated), "df"), 1L + as.integer(estimate))
    expect_true(estimated$power > 1 && estimated$power < 2)
    if (!estimate) expect_equal(estimated$power, 1.4)
  }
  expect_error(
    spglm(
      y ~ z,
      "tweedie",
      setup$data,
      c("s1", "s2"),
      power = 2,
      training_iterations = 0L
    ),
    "power"
  )
  expect_error(
    spglm(y ~ z, "tweedie", setup$data, c("s1", "s2"), estimate_power = NA),
    "estimate_power"
  )
})
