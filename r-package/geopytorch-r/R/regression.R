#' Fit a Gaussian spatial linear model
#'
#' Fits a linear mean with spatially correlated Gaussian errors. Regression
#' coefficients are profiled by generalized least squares. The covariance
#' parameters are fitted by ML or REML using GPyTorch and torch.optim.LBFGS.
#'
#' @param formula A regression formula with a numeric response. Offsets are not supported.
#' @param data A data frame containing the response and covariates.
#' @param coords Coordinate column names in data, or a numeric coordinate matrix.
#' @param covariance_model Spatial covariance model.
#' @param matern_nu Fixed Matérn smoothness: 0.5, 1.5, or 2.5.
#' @param initial Initial or fixed c(partial_sill, phi, nugget). NULL derives values from the data.
#' @param likelihood_method reml (default) or ml.
#' @param training_iterations Maximum optimizer iterations. Zero keeps initial parameters.
#' @param starts Optional list of positive initial parameter triples for multiple starts.
#' @param parameter_bounds Optional list of three c(lower, upper) pairs in physical units.
#'   Optimization starts must be strictly inside the bounds. GPyTorch constraints
#'   enforce bounds by transforming the raw parameters; boundary optima are approached
#'   from the interior. Fixed parameters may lie on bounds.
#'   Zero lower bounds allow parameters to approach zero. Python's default is unbounded.
#' @param device PyTorch device, default cpu.
#' @param dtype float64 (default) or float32.
#' @return A geopytorch_splm object with coefficients, their
#'   covariance, covariance parameters, likelihoods, optimizer diagnostics, and
#'   the retained Python model. Factor levels and contrasts are retained for prediction.
#' @details The range phi gives exponential correlation exp(-h/phi), Gaussian
#'   correlation exp(-(h/phi)^2), and Matérn Bessel argument h/phi at distance h.
#'   Prediction defaults
#'   to the latent process and includes regression-coefficient uncertainty,
#'   conditional on estimated covariance parameters. Computation is exact and dense.
#'   Missing values and rank-deficient designs are rejected. A zero nugget is
#'   supported when covariance parameters are fixed. logLik() uses the fitted
#'   likelihood method; REML omits the design-only term 0.5 * log(det(t(X) %*% X)).
#' @export
splm <- function(
  formula,
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
  likelihood_method = c("reml", "ml"),
  training_iterations = 600L,
  starts = NULL,
  parameter_bounds = NULL,
  device = "cpu",
  dtype = c("float64", "float32")
) {
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
  if (!is.null(stats::model.offset(frame))) {
    stop("formula offsets are not supported", call. = FALSE)
  }
  terms <- stats::terms(frame)
  X <- stats::model.matrix(terms, frame)
  y <- .geopytorch_values(stats::model.response(frame), nrow(coords))
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
  model <- geopytorch_module()$SpatialLinearModel(
    covariance_model,
    matern_nu = matern_nu
  )
  model$fit(
    tensor(coords),
    tensor(y),
    tensor(X),
    likelihood_method = likelihood_method,
    initial = initial,
    training_iterations = training_iterations,
    starts = starts,
    parameter_bounds = parameter_bounds
  )
  coefficients <- setNames(
    as.numeric(.geopytorch_tensor_to_r(model$coef_)),
    colnames(X)
  )
  covariance <- .geopytorch_tensor_to_r(model$coef_covariance_)
  dimnames(covariance) <- list(colnames(X), colnames(X))
  structure(
    list(
      call = match.call(),
      model = model,
      coefficients = coefficients,
      vcov = covariance,
      covariance_parameters = setNames(
        as.numeric(.geopytorch_tensor_to_r(model$covariance_parameters_)),
        c("partial_sill", "phi", "nugget")
      ),
      log_likelihood = as.numeric(.geopytorch_tensor_to_r(
        model$log_likelihood_
      )),
      restricted_log_likelihood = as.numeric(.geopytorch_tensor_to_r(
        model$restricted_log_likelihood_
      )),
      reml_design_constant = as.numeric(.geopytorch_tensor_to_r(
        model$reml_design_constant_
      )),
      converged = reticulate::py_to_r(model$converged_),
      optimizer_results = reticulate::py_to_r(model$optimizer_results_),
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
      covariance_estimated = training_iterations > 0L,
      device = device,
      dtype = dtype
    ),
    class = "geopytorch_splm"
  )
}

.geopytorch_regression_inputs <- function(object, newdata, newcoords = NULL) {
  if (!is.data.frame(newdata)) {
    stop("newdata must be a data frame", call. = FALSE)
  }
  terms <- stats::delete.response(object$terms)
  frame <- stats::model.frame(
    terms,
    newdata,
    na.action = stats::na.fail,
    xlev = object$xlevels
  )
  X <- stats::model.matrix(terms, frame, contrasts.arg = object$contrasts)
  if (!identical(colnames(X), object$design_columns)) {
    stop(
      "prediction design columns differ from the fitted design",
      call. = FALSE
    )
  }
  if (is.null(newcoords)) {
    if (is.null(object$coords_names)) {
      stop(
        "supply newcoords when training coordinates were a matrix",
        call. = FALSE
      )
    }
    if (!all(object$coords_names %in% names(newdata))) {
      stop("newdata is missing coordinate columns", call. = FALSE)
    }
    newcoords <- newdata[, object$coords_names, drop = FALSE]
  }
  newcoords <- .geopytorch_matrix(
    newcoords,
    "newcoords",
    object$input_dimension
  )
  if (nrow(X) != nrow(newcoords)) {
    stop(
      "prediction coordinates and design must have matching rows",
      call. = FALSE
    )
  }
  torch <- .geopytorch_torch()
  tensor <- function(x) {
    .geopytorch_tensor(
      x,
      .geopytorch_dtype(object$dtype, torch),
      object$device,
      torch
    )
  }
  list(
    coords = tensor(newcoords),
    design = tensor(X),
    offset = if (is.null(stats::model.offset(frame))) {
      NULL
    } else {
      tensor(stats::model.offset(frame))
    }
  )
}

#' Predict from a spatial linear model
#' @param object An object returned by splm().
#' @param newdata A data frame containing prediction covariates and coordinate columns.
#' @param newcoords Optional coordinate matrix; required if coords was a matrix during fitting.
#' @param observation_noise Include independent future nugget variance.
#' @param full_covariance Return the joint prediction covariance in R.
#' @param ... Unused.
#' @return A geopytorch_prediction object. Uncertainty includes estimation of
#'   regression coefficients but not estimation of covariance parameters.
#' @export
predict.geopytorch_splm <- function(
  object,
  newdata,
  newcoords = NULL,
  observation_noise = FALSE,
  full_covariance = FALSE,
  ...
) {
  observation_noise <- .geopytorch_flag(observation_noise, "observation_noise")
  full_covariance <- .geopytorch_flag(full_covariance, "full_covariance")
  inputs <- .geopytorch_regression_inputs(object, newdata, newcoords)
  prediction <- object$model$predict(
    inputs$coords,
    inputs$design,
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
      observation_noise = observation_noise
    ),
    class = "geopytorch_prediction"
  )
}

#' @export
coef.geopytorch_splm <- function(object, ...) {
  object$coefficients
}

#' @export
vcov.geopytorch_splm <- function(object, ...) object$vcov

#' @export
logLik.geopytorch_splm <- function(object, ...) {
  reml <- identical(object$likelihood_method, "reml")
  value <- if (reml) object$restricted_log_likelihood else object$log_likelihood
  # ML counts p estimated coefficients plus estimated covariance parameters;
  # REML counts covariance parameters only. Fixed covariance fits count none.
  df <- (if (object$covariance_estimated) 3L else 0L) +
    (if (reml) 0L else length(coef(object)))
  structure(
    value,
    class = "logLik",
    df = df,
    nobs = object$observations - if (reml) length(coef(object)) else 0L
  )
}

#' @export
print.geopytorch_splm <- function(x, ...) {
  cat("GeoPyTorch spatial linear model\n")
  cat(sprintf(
    "  observations: %d; coefficients: %d; method: %s\n",
    x$observations,
    length(x$coefficients),
    toupper(x$likelihood_method)
  ))
  print(x$coefficients)
  cat("Covariance parameters:\n")
  print(x$covariance_parameters)
  invisible(x)
}
