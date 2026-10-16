#' Fit a spatial generalized linear model
#'
#' Fits a regression with Gaussian spatial effects on the link scale. Covariance
#' and dispersion parameters maximize a dense Laplace approximation to the ML or
#' REML criterion. All observations are used together. The spatial nugget is distinct from
#' conditional response variation.
#'
#' @param formula Regression formula, optionally including offset(). A binomial
#'   response can be binary, a two-level factor, success counts with trials, or
#'   cbind(successes, failures).
#' @param family One of poisson, nbinomial, binomial, beta, Gamma, inverse.gaussian, tweedie.
#' @param data A data frame containing the response and covariates.
#' @param coords Coordinate column names or a numeric coordinate matrix.
#' @param covariance_model Spatial covariance model, as in splm().
#' @param matern_nu Fixed Matérn smoothness: 0.5, 1.5, or 2.5.
#' @param initial Initial or fixed c(partial_sill, phi, nugget).
#' @param dispersion Positive response dispersion, initially one.
#' @param estimate_dispersion Estimate dispersion when the family permits it.
#' @param power Initial or fixed Tweedie power, default 1.5.
#' @param estimate_power Estimate Tweedie power, default TRUE. Ignored for other families.
#' @param power_bounds Lower and upper bounds when estimating Tweedie power,
#'   default c(1.1, 1.999). Fixed power can be anywhere strictly between 1 and 2.
#' @param max_series_terms Maximum number of terms in the adaptive Tweedie density
#'   series. Failure to meet its tail tolerance reports a numerical failure.
#' @param fixed Three logical values specifying fixed covariance parameters.
#' @param trials Positive integer totals for a binomial count response.
#' @param likelihood_method reml (default) or ml, using the Laplace criteria below.
#' @param training_iterations Maximum outer optimizer iterations. Zero fixes
#'   covariance, dispersion, and power while still fitting the latent mode and coefficients.
#' @param starts Optional list of initial covariance triples. Fixed entries must agree.
#' @param parameter_bounds Bounds for the three covariance parameters, as in splm().
#' @param mode_iterations Maximum Newton iterations for each latent mode solve
#'   and, with ml, for the coefficient solve.
#' @param mode_tolerance Positive latent score tolerance, or NULL for a
#'   precision-dependent default. With ml, the coefficient solve stops when its
#'   squared Newton decrement is below the squared tolerance.
#' @param device PyTorch device, default cpu.
#' @param dtype float64 (default) or float32.
#' @return A geopytorch_spglm object supporting coef(), vcov(), fitted(),
#'   residuals(), logLik(), summary(), and predict().
#' @details All families use a log link except binomial and beta, which use logit.
#'   Dispersion is fixed at one for Poisson and binomial. Otherwise it denotes
#'   negative-binomial size, beta precision, Gamma shape, or inverse-Gaussian
#'   shape divided by the mean. Thus inverse-Gaussian variance is mu^2/dispersion.
#'   Tweedie responses may be zero or positive continuous values. Their mean is
#'   exp(eta) and variance is dispersion * mean^power, with 1 < power < 2.
#'   Tweedie density calculations and derivatives use PyTorch on the selected device.
#'   Responses with no finite coefficient estimate are rejected, for example
#'   all-zero counts with an intercept or binomial outcomes separated by the design.
#'   Covariance conventions match splm(). Missing values and deficient designs
#'   are rejected. Computation is dense. The ml criterion is the marginal Laplace
#'   log likelihood: the latent predictor is integrated at fixed coefficients,
#'   which then maximize it, and vcov() is its inverse negative Hessian in the
#'   coefficients. The reml criterion is a joint Laplace integral over the
#'   predictor and coefficients, minus n * log(2*pi) / 2; its coefficients are
#'   joint modes. logLik() reports the criterion of the fitted method. See the
#'   Python class documentation for the formulas. These approximate criteria
#'   would be exact for a Gaussian response, but they are not ordinary glm()
#'   log likelihoods.
#' @export
spglm <- function(
  formula,
  family,
  data,
  coords,
  covariance_model = c(
    "exponential",
    "matern",
    "gaussian",
    "spherical",
    "wave"
  ),
  matern_nu = 1.5,
  initial = NULL,
  dispersion = 1,
  estimate_dispersion = TRUE,
  fixed = NULL,
  trials = NULL,
  likelihood_method = c("reml", "ml"),
  training_iterations = 200L,
  starts = NULL,
  parameter_bounds = NULL,
  mode_iterations = 100L,
  mode_tolerance = NULL,
  device = "cpu",
  dtype = c("float64", "float32"),
  power = 1.5,
  estimate_power = TRUE,
  power_bounds = c(1.1, 1.999),
  max_series_terms = 16385L
) {
  family <- match.arg(
    family,
    c(
      "poisson",
      "nbinomial",
      "binomial",
      "beta",
      "Gamma",
      "inverse.gaussian",
      "tweedie"
    )
  )
  covariance_model <- match.arg(covariance_model)
  likelihood_method <- match.arg(likelihood_method)
  dtype <- match.arg(dtype)
  if (!inherits(formula, "formula")) {
    stop("formula must be a regression formula", call. = FALSE)
  }
  if (!is.data.frame(data)) {
    stop("data must be a data frame", call. = FALSE)
  }
  training_iterations <- .geopytorch_nonnegative_integer(
    training_iterations,
    "training_iterations"
  )
  mode_iterations <- .geopytorch_nonnegative_integer(
    mode_iterations,
    "mode_iterations"
  )
  estimate_dispersion <- .geopytorch_flag(
    estimate_dispersion,
    "estimate_dispersion"
  )
  if (
    !is.null(fixed) &&
      (!is.logical(fixed) || length(fixed) != 3L || anyNA(fixed))
  ) {
    stop("fixed must contain three logical values", call. = FALSE)
  }
  estimate_power <- .geopytorch_flag(estimate_power, "estimate_power")
  max_series_terms <- .geopytorch_nonnegative_integer(
    max_series_terms,
    "max_series_terms"
  )
  coords_names <- if (is.character(coords)) coords else NULL
  if (!is.null(coords_names)) {
    if (
      !length(coords_names) ||
        anyNA(coords_names) ||
        !all(coords_names %in% names(data))
    ) {
      stop("coords must name coordinate columns in data", call. = FALSE)
    }
    coords <- data[, coords_names, drop = FALSE]
  }
  coords <- .geopytorch_matrix(coords, "coords")
  frame <- stats::model.frame(formula, data, na.action = stats::na.fail)
  terms <- stats::terms(frame)
  X <- stats::model.matrix(terms, frame)
  y <- stats::model.response(frame)
  if (family == "binomial") {
    if (is.matrix(y)) {
      if (
        ncol(y) != 2L ||
          !is.numeric(y) ||
          any(!is.finite(y)) ||
          any(y < 0) ||
          any(y != round(y))
      ) {
        stop(
          "binomial matrix response must contain integer successes and failures",
          call. = FALSE
        )
      }
      if (!is.null(trials)) {
        stop("do not supply trials with a matrix response", call. = FALSE)
      }
      trials <- rowSums(y)
      y <- y[, 1L]
    } else if (is.factor(y)) {
      if (nlevels(y) != 2L || !is.null(trials)) {
        stop(
          "binomial factors must have two levels and no trials argument",
          call. = FALSE
        )
      }
      y <- as.numeric(y != levels(y)[1L])
    }
  }
  y <- .geopytorch_values(y, nrow(coords))
  if (nrow(X) != nrow(coords) || ncol(X) < 1L || nrow(X) <= ncol(X)) {
    stop(
      "coordinates and design must have matching rows and n > p > 0",
      call. = FALSE
    )
  }
  if (qr(X)$rank != ncol(X)) {
    stop("design matrix must have full column rank", call. = FALSE)
  }
  torch <- .geopytorch_torch()
  device <- .geopytorch_resolve_device(device, dtype, torch)
  torch_dtype <- .geopytorch_dtype(dtype, torch)
  tensor <- function(x) .geopytorch_tensor(x, torch_dtype, device, torch)
  model <- geopytorch_module()$SpatialGeneralizedLinearModel(
    family,
    covariance_model,
    matern_nu = matern_nu
  )
  model$fit(
    tensor(coords),
    tensor(y),
    tensor(X),
    likelihood_method = likelihood_method,
    initial = initial,
    dispersion = dispersion,
    estimate_dispersion = estimate_dispersion,
    fixed = if (is.null(fixed)) NULL else as.list(fixed),
    trials = if (is.null(trials)) NULL else tensor(trials),
    offset = if (is.null(stats::model.offset(frame))) {
      NULL
    } else {
      tensor(stats::model.offset(frame))
    },
    training_iterations = training_iterations,
    starts = starts,
    parameter_bounds = parameter_bounds,
    mode_iterations = mode_iterations,
    mode_tolerance = mode_tolerance,
    power = power,
    estimate_power = estimate_power,
    power_bounds = as.list(power_bounds),
    max_series_terms = max_series_terms
  )
  coefficients <- setNames(
    as.numeric(.geopytorch_tensor_to_r(model$coef_)),
    colnames(X)
  )
  covariance <- .geopytorch_tensor_to_r(model$coef_covariance_)
  dimnames(covariance) <- list(colnames(X), colnames(X))
  scalar <- function(x) as.numeric(.geopytorch_tensor_to_r(x))
  structure(
    list(
      call = match.call(),
      model = model,
      coefficients = coefficients,
      vcov = covariance,
      covariance_parameters = setNames(
        scalar(model$covariance_parameters_),
        c("partial_sill", "phi", "nugget")
      ),
      dispersion = scalar(model$dispersion_),
      power = if (family == "tweedie") scalar(model$power_) else NULL,
      power_estimated = reticulate::py_to_r(model$power_estimated_),
      family = family,
      log_likelihood = scalar(model$log_likelihood_),
      restricted_log_likelihood = scalar(model$restricted_log_likelihood_),
      fitted_values = scalar(model$fitted_values_),
      fitted_link = scalar(model$fitted_link_),
      trials = scalar(model$trials_),
      offset = scalar(model$offset_),
      converged = reticulate::py_to_r(model$converged_),
      optimizer_results = reticulate::py_to_r(model$optimizer_results_),
      mode_diagnostics = reticulate::py_to_r(model$mode_diagnostics_),
      terms = terms,
      xlevels = stats::.getXlevels(terms, frame),
      contrasts = attr(X, "contrasts"),
      design_columns = colnames(X),
      design_matrix = X,
      response = y,
      coords_names = coords_names,
      input_dimension = ncol(coords),
      observations = nrow(X),
      covariance_model = covariance_model,
      likelihood_method = likelihood_method,
      covariance_estimated = as.integer(reticulate::py_to_r(
        model$covariance_estimated_
      )),
      dispersion_estimated = reticulate::py_to_r(model$dispersion_estimated_),
      device = device,
      dtype = dtype
    ),
    class = "geopytorch_spglm"
  )
}

#' Predict from a spatial generalized linear model
#' @param object An object returned by spglm().
#' @param newdata Prediction covariates and coordinate columns. Formula offsets are evaluated here.
#' @param newcoords Optional coordinate matrix, required if training coords was a matrix.
#' @param type response (default) or link.
#' @param trials Binomial totals at prediction locations, default one.
#' @param observation_noise Include conditional response variance, on the response scale only.
#' @param full_covariance Return the joint prediction covariance in R.
#' @param ... Unused.
#' @return A geopytorch_prediction object. Link predictions include a new independent
#'   nugget and account for coefficients and latent-mode uncertainty. Response
#'   means use the inverse link and variances use the delta method. They are not
#'   integrated response means. Covariance and dispersion estimation uncertainty
#'   is excluded. Use fitted() to obtain training fitted values.
#' @export
predict.geopytorch_spglm <- function(
  object,
  newdata,
  newcoords = NULL,
  type = c("response", "link"),
  trials = NULL,
  observation_noise = FALSE,
  full_covariance = FALSE,
  ...
) {
  type <- match.arg(type)
  observation_noise <- .geopytorch_flag(observation_noise, "observation_noise")
  full_covariance <- .geopytorch_flag(full_covariance, "full_covariance")
  inputs <- .geopytorch_regression_inputs(object, newdata, newcoords)
  torch <- .geopytorch_torch()
  totals <- if (is.null(trials)) {
    NULL
  } else {
    .geopytorch_tensor(
      trials,
      .geopytorch_dtype(object$dtype, torch),
      object$device,
      torch
    )
  }
  prediction <- object$model$predict(
    inputs$coords,
    inputs$design,
    type = type,
    trials = totals,
    offset = inputs$offset,
    observation_noise = observation_noise
  )
  variance <- as.numeric(.geopytorch_tensor_to_r(prediction$variance))
  structure(
    list(
      mean = as.numeric(.geopytorch_tensor_to_r(prediction$mean)),
      variance = variance,
      standard_deviation = sqrt(pmax(variance, 0)),
      covariance_matrix = if (full_covariance) {
        .geopytorch_tensor_to_r(prediction$covariance_matrix)
      } else {
        NULL
      },
      device = object$device,
      dtype = object$dtype,
      type = type,
      observation_noise = observation_noise
    ),
    class = "geopytorch_prediction"
  )
}

#' @export
coef.geopytorch_spglm <- function(object, ...) object$coefficients

#' @export
vcov.geopytorch_spglm <- function(object, ...) object$vcov

#' @export
fitted.geopytorch_spglm <- function(object, type = c("response", "link"), ...) {
  if (match.arg(type) == "response") {
    object$fitted_values
  } else {
    object$fitted_link
  }
}

#' @export
residuals.geopytorch_spglm <- function(object, ...) {
  object$response - object$fitted_values
}

#' @export
nobs.geopytorch_spglm <- function(object, ...) object$observations

#' @export
logLik.geopytorch_spglm <- function(object, ...) {
  reml <- identical(object$likelihood_method, "reml")
  value <- if (reml) object$restricted_log_likelihood else object$log_likelihood
  df <- object$covariance_estimated +
    as.integer(object$dispersion_estimated) +
    as.integer(isTRUE(object$power_estimated)) +
    if (reml) 0L else length(coef(object))
  structure(value, class = "logLik", df = df, nobs = object$observations)
}

#' @export
summary.geopytorch_spglm <- function(object, ...) {
  se <- sqrt(diag(vcov(object)))
  z <- coef(object) / se
  structure(
    list(
      call = object$call,
      family = object$family,
      coefficients = cbind(
        Estimate = coef(object),
        "Std. Error" = se,
        "z value" = z,
        "Pr(>|z|)" = 2 * stats::pnorm(-abs(z))
      ),
      covariance_parameters = object$covariance_parameters,
      dispersion = object$dispersion,
      power = object$power,
      log_likelihood = logLik(object),
      converged = object$converged
    ),
    class = "summary.geopytorch_spglm"
  )
}

#' @export
print.geopytorch_spglm <- function(x, ...) {
  cat(sprintf(
    "GeoPyTorch spatial GLM (%s, %s)\n",
    x$family,
    toupper(x$likelihood_method)
  ))
  print(x$coefficients)
  cat("Covariance parameters:\n")
  print(x$covariance_parameters)
  cat(sprintf("Dispersion: %g; converged: %s\n", x$dispersion, x$converged))
  if (!is.null(x$power)) {
    cat(sprintf("Tweedie power: %g\n", x$power))
  }
  invisible(x)
}

#' @export
print.summary.geopytorch_spglm <- function(x, ...) {
  cat(sprintf("GeoPyTorch spatial GLM (%s)\n", x$family))
  stats::printCoefmat(x$coefficients)
  cat("Covariance parameters:\n")
  print(x$covariance_parameters)
  cat(sprintf("Dispersion: %g; converged: %s\n", x$dispersion, x$converged))
  if (!is.null(x$power)) {
    cat(sprintf("Tweedie power: %g\n", x$power))
  }
  invisible(x)
}
