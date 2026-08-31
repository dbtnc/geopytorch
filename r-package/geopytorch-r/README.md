# GeoPyTorch-R

'GeoPyTorch-R' is the R interface to the Python GeoPyTorch package. Its formal
R package identifier is 'geopytorch', so it is loaded with:

~~~r
library(geopytorch)
~~~

The Python module is imported lazily. Loading the R package declares the Python
requirement but does not initialize Python.

## Ordinary Kriging

CPU execution and double precision are the defaults:

~~~r
fit <- ordinary_kriging(
  train_x = training_coordinates,
  train_y = training_values,
  covariance_model = "exponential",
  partial_sill = 1,
  phi = 0.2,
  nugget = 0.05,
  training_iterations = 0
)

prediction <- predict(
  fit,
  target_coordinates,
  observation_noise = TRUE
)
prediction$mean
prediction$variance
~~~

Select a CUDA GPU by changing one argument:

~~~r
fit <- ordinary_kriging(
  training_coordinates,
  training_values,
  device = "cuda"
)
~~~

Use 'geopytorch_devices()' to list available devices. 'device = "auto"'
prefers CUDA, then Apple MPS when 'dtype = "float32"', and otherwise CPU.
Apple MPS does not support the package's default double precision, so request
it with 'device = "mps", dtype = "float32"'. CPU thread counts can be set
before fitting with 'geopytorch_threads(threads = 8, interop_threads = 1)'.
