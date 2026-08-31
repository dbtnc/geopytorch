test_that("package loading leaves the Python import delayed", {
  skip_if(
    reticulate::py_available(initialize = FALSE),
    "the test harness initialized Python before this test"
  )

  module <- geopytorch_module()

  expect_s3_class(module, "python.builtin.module")
  expect_false(reticulate::py_available(initialize = FALSE))
})

test_that("the requirement is registered without initializing Python", {
  requirement <- geopytorch_requirement()
  manifest <- reticulate::py_require()

  expect_length(requirement, 1L)
  expect_true(nzchar(requirement))
  expect_true(requirement %in% manifest$packages)
})

test_that("a local requirement override is honored", {
  override <- Sys.getenv("GEOPYTORCH_PYTHON_REQUIREMENT", unset = "")
  skip_if(!nzchar(override), "no local requirement override was requested")

  expect_identical(geopytorch_requirement(), trimws(override))
})
