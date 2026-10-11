# Purpose: Generate independent spatial GLM reference fits for Python tests.
# Inputs: spmodel 0.14.0 or later and jsonlite; run from the project root.
# Outputs: tests/fixtures/spglm_reference.json with fixed and estimated fits.
# Research log: 003_geopytorch_spglm_implementation.tex in the research session.

stopifnot(requireNamespace("spmodel"), requireNamespace("jsonlite"))
set.seed(417)
n <- 18L
d <- data.frame(
  east = runif(n),
  north = runif(n),
  z = seq(-1, 1, length.out = n)
)
d$exposure <- rep(c(0.8, 1, 1.3), length.out = n)
new <- data.frame(
  east = c(.12, .53, .88),
  north = c(.21, .44, .73),
  z = c(-.6, .1, .7),
  exposure = c(.9, 1.1, 1.2)
)
families <- c(
  "poisson",
  "nbinomial",
  "binomial",
  "beta",
  "Gamma",
  "inverse.gaussian"
)
responses <- list(
  poisson = c(1, 4, 0, 2, 3, 1, 6, 2, 1, 5, 4, 2, 7, 3, 4, 8, 2, 5),
  nbinomial = c(0, 4, 1, 2, 6, 0, 3, 2, 7, 1, 5, 10, 2, 4, 12, 3, 6, 9),
  binomial = c(1, 5, 2, 4, 3, 6, 2, 5, 3, 7, 4, 6, 8, 5, 7, 6, 9, 7),
  beta = c(
    .12,
    .47,
    .23,
    .38,
    .61,
    .29,
    .35,
    .7,
    .48,
    .24,
    .66,
    .55,
    .81,
    .43,
    .73,
    .62,
    .9,
    .76
  ),
  Gamma = c(
    .4,
    1.8,
    .7,
    1.1,
    2.3,
    .6,
    1.5,
    3.1,
    .8,
    2.5,
    1.6,
    4,
    2.2,
    1.2,
    3.8,
    2.9,
    4.5,
    1.9
  ),
  inverse.gaussian = c(
    .4,
    1.8,
    .7,
    1.1,
    2.3,
    .6,
    1.5,
    3.1,
    .8,
    2.5,
    1.6,
    4,
    2.2,
    1.2,
    3.8,
    2.9,
    4.5,
    1.9
  )
)
dispersions <- c(
  poisson = 1,
  nbinomial = 4,
  binomial = 1,
  beta = 7,
  Gamma = 3,
  inverse.gaussian = 2
)
models <- c("exponential", "gaussian", "matern", "spherical", "wave")
results <- list()
for (family in families) {
  d$y <- responses[[family]]
  d$total <- 10 + seq_len(n) %% 3
  d$failure <- d$total - d$y
  formula <- if (family == "binomial") {
    cbind(y, failure) ~ z + offset(log(exposure))
  } else {
    y ~ z + offset(log(exposure))
  }
  for (model in models) {
    for (method in c("ml", "reml")) {
      theta <- c(.7, .3, .2)
      spcov <- list(
        spcov_type = model,
        de = theta[1],
        ie = theta[3],
        range = theta[2] * if (model == "matern") sqrt(3) else 1,
        known = c("de", "ie", "range")
      )
      if (model == "matern") {
        spcov$extra <- 1.5
        spcov$known <- c(spcov$known, "extra")
      }
      fit <- spmodel::spglm(
        formula,
        data = d,
        xcoord = "east",
        ycoord = "north",
        estmethod = method,
        local = FALSE,
        spcov_initial = do.call(spmodel::spcov_initial, spcov),
        dispersion_initial = do.call(
          spmodel::dispersion_initial,
          list(
            family = family,
            dispersion = dispersions[[family]],
            known = "dispersion"
          )
        ),
        diagtol = 0
      )
      link <- predict(fit, new, type = "link", se.fit = TRUE, local = FALSE)
      response <- predict(
        fit,
        new,
        type = "response",
        se.fit = TRUE,
        delta = TRUE,
        local = FALSE,
        newdata_size = if (family == "binomial") c(5, 8, 12) else NULL
      )
      results[[length(results) + 1L]] <- list(
        family = family,
        covariance_model = model,
        method = method,
        initial = theta,
        dispersion = dispersions[[family]],
        y = d$y,
        trials = if (family == "binomial") d$total else NULL,
        coefficients = unname(coef(fit)),
        vcov = unname(vcov(fit)),
        log_likelihood = as.numeric(logLik(fit)),
        fitted_link = unname(fitted(fit, type = "link")),
        fitted_response = unname(fitted(fit)),
        prediction_link = unname(link$fit),
        variance_link = unname(link$se.fit^2),
        prediction_response = unname(response$fit),
        variance_response = unname(response$se.fit^2)
      )
    }
  }
}
estimated <- list()
for (family in families) {
  d$y <- responses[[family]]
  d$failure <- d$total - d$y
  formula <- if (family == "binomial") {
    cbind(y, failure) ~ z + offset(log(exposure))
  } else {
    y ~ z + offset(log(exposure))
  }
  for (method in c("ml", "reml")) {
    # Fix range and nugget so boundary variance estimates remain identifiable.
    fit <- spmodel::spglm(
      formula,
      data = d,
      xcoord = "east",
      ycoord = "north",
      estmethod = method,
      local = FALSE,
      spcov_initial = spmodel::spcov_initial(
        "exponential",
        de = .7,
        ie = .2,
        range = .3,
        known = c("ie", "range")
      ),
      dispersion_initial = do.call(
        spmodel::dispersion_initial,
        list(
          family = family,
          dispersion = dispersions[[family]],
          known = if (family %in% c("poisson", "binomial")) {
            "dispersion"
          } else {
            character()
          }
        )
      ),
      diagtol = 0,
      control = list(maxit = 1000, reltol = 1e-10)
    )
    estimated[[length(estimated) + 1L]] <- list(
      family = family,
      covariance_model = "exponential",
      method = method,
      initial = c(.7, .3, .2),
      dispersion = dispersions[[family]],
      y = d$y,
      trials = if (family == "binomial") d$total else NULL,
      coefficients = unname(coef(fit)),
      vcov = unname(vcov(fit)),
      parameters = as.numeric(fit$coefficients$spcov[c("de", "range", "ie")]),
      estimated_dispersion = as.numeric(fit$coefficients$dispersion),
      log_likelihood = as.numeric(logLik(fit)),
      convergence = fit$optim$convergence
    )
  }
}
dispersion_fits <- list()
for (family in setdiff(families, c("poisson", "binomial"))) {
  d$y <- responses[[family]]
  theta <- c(.05, .3, .01)
  for (method in c("ml", "reml")) {
    fit_at <- function(phi, known = character()) {
      spmodel::spglm(
        y ~ z + offset(log(exposure)),
        data = d,
        xcoord = "east",
        ycoord = "north",
        estmethod = method,
        local = FALSE,
        spcov_initial = spmodel::spcov_initial(
          "exponential",
          de = theta[1],
          ie = theta[3],
          range = theta[2],
          known = c("de", "ie", "range")
        ),
        dispersion_initial = do.call(
          spmodel::dispersion_initial,
          list(family = family, dispersion = phi, known = known)
        ),
        diagtol = 0,
        control = list(maxit = 1000, reltol = 1e-10)
      )
    }
    if (family == "inverse.gaussian") {
      # Direct joint optimization fails for this example; optimize its fixed-parameter likelihood.
      optimum <- optimize(
        function(log_phi) {
          -as.numeric(logLik(fit_at(exp(log_phi), "dispersion")))
        },
        interval = log(c(.5, 20)),
        tol = 1e-9
      )
      fit <- fit_at(exp(optimum$minimum), "dispersion")
    } else {
      fit <- fit_at(dispersions[[family]])
    }
    dispersion_fits[[length(dispersion_fits) + 1L]] <- list(
      family = family,
      covariance_model = "exponential",
      method = method,
      initial = theta,
      dispersion = dispersions[[family]],
      y = d$y,
      trials = NULL,
      coefficients = unname(coef(fit)),
      vcov = unname(vcov(fit)),
      estimated_dispersion = as.numeric(fit$coefficients$dispersion),
      log_likelihood = as.numeric(logLik(fit)),
      reference_optimizer = if (family == "inverse.gaussian") {
        "stats::optimize of fixed-parameter spmodel::spglm"
      } else {
        "spmodel::spglm"
      }
    )
  }
}
dir.create("tests/fixtures", showWarnings = FALSE)
jsonlite::write_json(
  list(
    version = as.character(packageVersion("spmodel")),
    coords = unname(as.matrix(d[c("east", "north")])),
    design = unname(model.matrix(~z, d)),
    offset = log(d$exposure),
    newcoords = unname(as.matrix(new[c("east", "north")])),
    newdesign = unname(model.matrix(~z, new)),
    newoffset = log(new$exposure),
    newtrials = c(5, 8, 12),
    fits = results,
    estimated = estimated,
    dispersion_fits = dispersion_fits
  ),
  "tests/fixtures/spglm_reference.json",
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE,
  null = "null"
)
