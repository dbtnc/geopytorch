test_that("the selected Python module comes from the live checkout", {
  checkout <- Sys.getenv("GEOPYTORCH_LIVE_CHECKOUT", unset = "")
  skip_if(!nzchar(checkout), "no live checkout was requested")

  expected_source <- normalizePath(
    file.path(checkout, "src", "geopytorch"),
    mustWork = TRUE
  )
  module <- geopytorch_module()
  module_path <- normalizePath(
    reticulate::py_to_r(module[["__file__"]]),
    mustWork = TRUE
  )

  expect_true(startsWith(module_path, expected_source))
  expect_setequal(
    unlist(reticulate::py_to_r(module$COVARIANCE_MODELS)),
    c("gaussian", "exponential", "matern", "spherical", "wave")
  )

  kernel <- module$covariance_kernel(
    "exponential",
    ard_num_dims = as.integer(2)
  )
  expect_true(reticulate::py_has_attr(kernel, "base_kernel"))

  configuration <- geopytorch_config()
  expect_s3_class(configuration, "geopytorch_config")
  expect_identical(
    normalizePath(configuration$module_path, mustWork = TRUE),
    module_path
  )
  expect_false(anyNA(
    unlist(configuration[c("geopytorch", "torch", "gpytorch", "numpy")])
  ))
})
