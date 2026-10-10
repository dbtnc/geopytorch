.geopytorch_torch <- function() {
  reticulate::import("torch", convert = FALSE)
}

.geopytorch_gpytorch <- function() {
  reticulate::import("gpytorch", convert = FALSE)
}

.geopytorch_python_scalar <- function(value) {
  reticulate::py_to_r(value)
}

.geopytorch_scalar_number <- function(
  value,
  name,
  lower = -Inf,
  strict = FALSE
) {
  if (
    length(value) != 1L ||
      !is.numeric(value) ||
      is.na(value) ||
      !is.finite(value)
  ) {
    stop(sprintf("%s must be one finite number", name), call. = FALSE)
  }
  invalid <- if (strict) value <= lower else value < lower
  if (invalid) {
    comparison <- if (strict) "greater than" else "at least"
    stop(
      sprintf("%s must be %s %s", name, comparison, format(lower)),
      call. = FALSE
    )
  }
  as.numeric(value)
}

.geopytorch_positive_integer <- function(value, name) {
  number <- .geopytorch_scalar_number(value, name, lower = 1)
  if (number != floor(number)) {
    stop(sprintf("%s must be an integer", name), call. = FALSE)
  }
  as.integer(number)
}

.geopytorch_nonnegative_integer <- function(value, name) {
  number <- .geopytorch_scalar_number(value, name, lower = 0)
  if (number != floor(number)) {
    stop(sprintf("%s must be an integer", name), call. = FALSE)
  }
  as.integer(number)
}

.geopytorch_flag <- function(value, name) {
  if (length(value) != 1L || !is.logical(value) || is.na(value)) {
    stop(sprintf("%s must be TRUE or FALSE", name), call. = FALSE)
  }
  value
}

.geopytorch_matrix <- function(value, name, columns = NULL) {
  if (is.data.frame(value)) {
    numeric_columns <- vapply(value, is.numeric, logical(1))
    if (!all(numeric_columns)) {
      stop(sprintf("%s must contain only numeric columns", name), call. = FALSE)
    }
  }
  value <- as.matrix(value)
  if (!is.numeric(value) || length(dim(value)) != 2L) {
    stop(sprintf("%s must be a numeric matrix or data frame", name), call. = FALSE)
  }
  if (nrow(value) < 1L || ncol(value) < 1L) {
    stop(sprintf("%s must have at least one row and one column", name), call. = FALSE)
  }
  if (!is.null(columns) && ncol(value) != columns) {
    stop(
      sprintf("%s must have %d columns", name, columns),
      call. = FALSE
    )
  }
  storage.mode(value) <- "double"
  if (any(!is.finite(value))) {
    stop(sprintf("%s must contain only finite values", name), call. = FALSE)
  }
  value
}

.geopytorch_values <- function(value, expected_length) {
  if (is.matrix(value) && ncol(value) == 1L) {
    value <- value[, 1L]
  }
  if (!is.numeric(value) || !is.null(dim(value))) {
    stop("train_y must be a numeric vector", call. = FALSE)
  }
  value <- as.numeric(value)
  if (length(value) != expected_length) {
    stop("train_x and train_y must contain the same number of rows", call. = FALSE)
  }
  if (any(!is.finite(value))) {
    stop("train_y must contain only finite values", call. = FALSE)
  }
  value
}

.geopytorch_dtype <- function(dtype, torch) {
  if (identical(dtype, "float64")) torch$float64 else torch$float32
}

.geopytorch_cuda_available <- function(torch) {
  isTRUE(.geopytorch_python_scalar(torch$cuda$is_available()))
}

.geopytorch_mps_available <- function(torch) {
  reticulate::py_has_attr(torch$backends, "mps") &&
    isTRUE(.geopytorch_python_scalar(torch$backends$mps$is_available()))
}

.geopytorch_resolve_device <- function(device, dtype, torch) {
  if (length(device) != 1L || !is.character(device) || is.na(device)) {
    stop("device must be one character string", call. = FALSE)
  }
  device <- tolower(trimws(device))
  if (!nzchar(device)) {
    stop("device must not be empty", call. = FALSE)
  }

  if (identical(device, "auto")) {
    if (.geopytorch_cuda_available(torch)) {
      device <- "cuda:0"
    } else if (identical(dtype, "float32") && .geopytorch_mps_available(torch)) {
      device <- "mps"
    } else {
      device <- "cpu"
    }
  }

  if (identical(device, "cpu")) {
    return(device)
  }
  if (identical(device, "mps")) {
    if (!.geopytorch_mps_available(torch)) {
      stop("MPS is unavailable in the selected PyTorch runtime", call. = FALSE)
    }
    if (!identical(dtype, "float32")) {
      stop("MPS requires dtype = \"float32\"", call. = FALSE)
    }
    return(device)
  }
  if (!grepl("^cuda(:[0-9]+)?$", device)) {
    stop(
      "device must be \"cpu\", \"auto\", \"mps\", \"cuda\", or \"cuda:N\"",
      call. = FALSE
    )
  }
  if (!.geopytorch_cuda_available(torch)) {
    stop("CUDA is unavailable in the selected PyTorch runtime", call. = FALSE)
  }
  if (grepl(":", device, fixed = TRUE)) {
    index <- as.integer(sub("^cuda:", "", device))
    count <- as.integer(.geopytorch_python_scalar(torch$cuda$device_count()))
    if (index >= count) {
      stop(
        sprintf("device %s was requested, but only %d CUDA devices are visible", device, count),
        call. = FALSE
      )
    }
  }
  device
}

#' List PyTorch devices available to GeoPyTorch
#'
#' Initializes Python and returns the device strings accepted by
#' ordinary_kriging(). CPU is always included.
#'
#' @return A character vector of available devices.
#' @export
geopytorch_devices <- function() {
  torch <- .geopytorch_torch()
  devices <- "cpu"
  if (.geopytorch_cuda_available(torch)) {
    count <- as.integer(.geopytorch_python_scalar(torch$cuda$device_count()))
    devices <- c(devices, paste0("cuda:", seq.int(0L, count - 1L)))
  }
  if (.geopytorch_mps_available(torch)) {
    devices <- c(devices, "mps")
  }
  devices
}

#' Resolve and validate a PyTorch device
#'
#' @param device Device string. Auto prefers CUDA, then MPS when using
#'   single precision, then CPU.
#' @param dtype float64 or float32.
#' @return A validated device string.
#' @export
geopytorch_device <- function(device = "cpu", dtype = c("float64", "float32")) {
  dtype <- match.arg(dtype)
  .geopytorch_resolve_device(device, dtype, .geopytorch_torch())
}

#' Get or set PyTorch CPU thread counts
#'
#' Calling this function initializes Python. PyTorch permits changing its
#' inter-op thread count only before parallel work has started.
#'
#' @param threads Optional positive intra-op thread count.
#' @param interop_threads Optional positive inter-op thread count.
#' @return A list containing the effective thread counts.
#' @export
geopytorch_threads <- function(threads = NULL, interop_threads = NULL) {
  torch <- .geopytorch_torch()
  if (!is.null(threads)) {
    torch$set_num_threads(.geopytorch_positive_integer(threads, "threads"))
  }
  if (!is.null(interop_threads)) {
    torch$set_num_interop_threads(
      .geopytorch_positive_integer(interop_threads, "interop_threads")
    )
  }
  list(
    threads = as.integer(.geopytorch_python_scalar(torch$get_num_threads())),
    interop_threads = as.integer(
      .geopytorch_python_scalar(torch$get_num_interop_threads())
    )
  )
}

.geopytorch_tensor <- function(value, dtype, device, torch) {
  torch$tensor(value, dtype = dtype, device = device)
}

.geopytorch_tensor_to_r <- function(tensor) {
  reticulate::py_to_r(tensor$detach()$to(device = "cpu")$numpy())
}

#' Fit an Ordinary-Kriging model with GeoPyTorch
#'
#' CPU execution is the default. Select a GPU with device = "cuda" or
#' device = "cuda:N"; Apple GPU execution uses device = "mps" together
#' with dtype = "float32".
#'
#' @param train_x Numeric matrix or data frame of observation coordinates.
#' @param train_y Numeric vector of observed values.
#' @param covariance_model Covariance model name.
#' @param matern_nu Matérn smoothness, one of 0.5, 1.5, or 2.5.
#' @param ard Whether to estimate a separate range for each coordinate.
#' @param partial_sill Optional fixed or initial covariance partial sill.
#' @param phi Optional fixed or initial covariance range.
#' @param nugget Optional fixed or initial Gaussian noise variance.
#' @param nugget_lower_bound Strict lower bound used when nugget is supplied.
#' @param training_iterations Number of optimizer iterations. Zero retains the
#'   supplied or initialized covariance parameters.
#' @param learning_rate Positive optimizer learning rate.
#' @param likelihood_method ml or reml.
#' @param optimizer adam or lbfgs.
#' @param device PyTorch device; defaults to cpu.
#' @param dtype float64 or float32.
#' @return An object of class geopytorch_ordinary_kriging.
#' @export
ordinary_kriging <- function(
  train_x,
  train_y,
  covariance_model = c(
    "matern",
    "exponential",
    "gaussian",
    "spherical",
    "wave"
  ),
  matern_nu = 1.5,
  ard = FALSE,
  partial_sill = NULL,
  phi = NULL,
  nugget = NULL,
  nugget_lower_bound = 0,
  training_iterations = 100L,
  learning_rate = 0.1,
  likelihood_method = c("ml", "reml"),
  optimizer = c("adam", "lbfgs"),
  device = "cpu",
  dtype = c("float64", "float32")
) {
  covariance_model <- match.arg(covariance_model)
  likelihood_method <- match.arg(likelihood_method)
  optimizer <- match.arg(optimizer)
  dtype <- match.arg(dtype)
  ard <- .geopytorch_flag(ard, "ard")
  matern_nu <- .geopytorch_scalar_number(matern_nu, "matern_nu")
  if (!matern_nu %in% c(0.5, 1.5, 2.5)) {
    stop("matern_nu must be one of 0.5, 1.5, or 2.5", call. = FALSE)
  }
  training_iterations <- .geopytorch_nonnegative_integer(
    training_iterations,
    "training_iterations"
  )
  learning_rate <- .geopytorch_scalar_number(
    learning_rate,
    "learning_rate",
    lower = 0,
    strict = TRUE
  )
  nugget_lower_bound <- .geopytorch_scalar_number(
    nugget_lower_bound,
    "nugget_lower_bound",
    lower = 0
  )
  if (!is.null(partial_sill)) {
    partial_sill <- .geopytorch_scalar_number(
      partial_sill,
      "partial_sill",
      lower = 0,
      strict = TRUE
    )
  }
  if (!is.null(phi)) {
    phi <- .geopytorch_scalar_number(phi, "phi", lower = 0, strict = TRUE)
  }
  if (!is.null(nugget)) {
    nugget <- .geopytorch_scalar_number(
      nugget,
      "nugget",
      lower = nugget_lower_bound,
      strict = TRUE
    )
  }

  train_x <- .geopytorch_matrix(train_x, "train_x")
  if (nrow(train_x) < 2L) {
    stop("at least two observations are required", call. = FALSE)
  }
  train_y <- .geopytorch_values(train_y, nrow(train_x))
  torch <- .geopytorch_torch()
  gpytorch <- .geopytorch_gpytorch()
  module <- geopytorch_module()
  device <- .geopytorch_resolve_device(device, dtype, torch)
  torch_dtype <- .geopytorch_dtype(dtype, torch)
  train_x_tensor <- .geopytorch_tensor(train_x, torch_dtype, device, torch)
  train_y_tensor <- .geopytorch_tensor(train_y, torch_dtype, device, torch)

  kernel <- NULL
  if (!is.null(partial_sill) || !is.null(phi)) {
    ard_num_dims <- if (ard) as.integer(ncol(train_x)) else NULL
    kernel <- module$covariance_kernel(
      covariance_model,
      matern_nu = matern_nu,
      ard_num_dims = ard_num_dims
    )
    kernel <- kernel$to(device = device, dtype = torch_dtype)
    if (!is.null(partial_sill)) {
      kernel$initialize(outputscale = partial_sill)
    }
    if (!is.null(phi)) {
      kernel$base_kernel$initialize(lengthscale = phi)
    }
  }

  likelihood <- NULL
  if (!is.null(nugget)) {
    constraint <- gpytorch$constraints$GreaterThan(nugget_lower_bound)
    likelihood <- gpytorch$likelihoods$GaussianLikelihood(
      noise_constraint = constraint
    )
    likelihood <- likelihood$to(device = device, dtype = torch_dtype)
    likelihood$initialize(noise = nugget)
  }

  model <- module$OrdinaryKriging(
    kernel = kernel,
    likelihood = likelihood,
    covariance_model = covariance_model,
    matern_nu = matern_nu,
    ard = ard
  )
  model$fit(
    train_x_tensor,
    train_y_tensor,
    training_iterations = training_iterations,
    learning_rate = learning_rate,
    likelihood_method = likelihood_method,
    optimizer = optimizer
  )

  structure(
    list(
      model = model,
      device = device,
      dtype = dtype,
      covariance_model = covariance_model,
      input_dimension = ncol(train_x),
      observations = nrow(train_x),
      training_iterations = training_iterations,
      likelihood_method = likelihood_method,
      optimizer = optimizer
    ),
    class = "geopytorch_ordinary_kriging"
  )
}

#' @export
print.geopytorch_ordinary_kriging <- function(x, ...) {
  cat("GeoPyTorch Ordinary-Kriging model\n")
  cat(sprintf("  observations    : %d\n", x$observations))
  cat(sprintf("  input dimensions: %d\n", x$input_dimension))
  cat(sprintf("  covariance model: %s\n", x$covariance_model))
  cat(sprintf("  device          : %s\n", x$device))
  cat(sprintf("  dtype           : %s\n", x$dtype))
  invisible(x)
}

#' Predict from a GeoPyTorch Ordinary-Kriging model
#'
#' @param object A fitted object returned by ordinary_kriging().
#' @param newdata Numeric matrix or data frame of prediction coordinates.
#' @param observation_noise Include the fitted nugget in prediction variance.
#' @param full_covariance Transfer the full joint covariance matrix to R.
#' @param ... Unused.
#' @return An object of class geopytorch_prediction containing mean,
#'   marginal variance, standard deviation, and optionally the joint covariance.
#' @export
predict.geopytorch_ordinary_kriging <- function(
  object,
  newdata,
  observation_noise = FALSE,
  full_covariance = FALSE,
  ...
) {
  observation_noise <- .geopytorch_flag(
    observation_noise,
    "observation_noise"
  )
  full_covariance <- .geopytorch_flag(full_covariance, "full_covariance")
  newdata <- .geopytorch_matrix(
    newdata,
    "newdata",
    columns = object$input_dimension
  )
  torch <- .geopytorch_torch()
  target <- .geopytorch_tensor(
    newdata,
    .geopytorch_dtype(object$dtype, torch),
    object$device,
    torch
  )
  prediction <- object$model$predict(
    target,
    observation_noise = observation_noise
  )
  mean <- as.numeric(.geopytorch_tensor_to_r(prediction$mean))
  variance <- as.numeric(.geopytorch_tensor_to_r(prediction$variance))
  covariance_matrix <- if (full_covariance) {
    .geopytorch_tensor_to_r(prediction$covariance_matrix)
  } else {
    NULL
  }

  structure(
    list(
      mean = mean,
      variance = variance,
      standard_deviation = sqrt(pmax(variance, 0)),
      covariance_matrix = covariance_matrix,
      device = object$device,
      dtype = object$dtype,
      observation_noise = observation_noise
    ),
    class = "geopytorch_prediction"
  )
}

#' @export
print.geopytorch_prediction <- function(x, ...) {
  cat("GeoPyTorch prediction\n")
  cat(sprintf("  locations: %d\n", length(x$mean)))
  cat(sprintf("  device   : %s\n", x$device))
  cat(sprintf("  dtype    : %s\n", x$dtype))
  invisible(x)
}

#' Extract Kriging weights
#'
#' @param object A fitted object returned by ordinary_kriging() or spatial_linear_model().
#' @param newdata Prediction coordinates for ordinary kriging; a prediction data frame for spatial linear models.
#' @param newcoords Optional prediction coordinate matrix for spatial linear models.
#' @return A numeric matrix with prediction locations in rows and observations
#'   in columns.
#' @export
kriging_weights <- function(object, newdata, newcoords = NULL) {
  if (inherits(object, "geopytorch_spatial_linear_model")) {
    inputs <- .geopytorch_regression_inputs(object, newdata, newcoords)
    return(.geopytorch_tensor_to_r(object$model$weights(inputs$coords, inputs$design)))
  }
  if (!inherits(object, "geopytorch_ordinary_kriging")) {
    stop("object must be returned by ordinary_kriging() or spatial_linear_model()", call. = FALSE)
  }
  newdata <- .geopytorch_matrix(
    newdata,
    "newdata",
    columns = object$input_dimension
  )
  torch <- .geopytorch_torch()
  target <- .geopytorch_tensor(
    newdata,
    .geopytorch_dtype(object$dtype, torch),
    object$device,
    torch
  )
  .geopytorch_tensor_to_r(object$model$weights(target))
}
