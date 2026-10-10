# GeoPyTorch-R

'GeoPyTorch-R' is the R interface to the Python GeoPyTorch package. Its formal
R package identifier is 'geopytorch', so it is loaded with:

~~~r
library(geopytorch)
~~~

The Python module is imported lazily. Loading the R package declares the Python
requirement but does not initialize Python.

## Spatial linear model

```r
fit <- spatial_linear_model(
  y ~ elevation + land_use,
  data = training_data,
  coords = c("east", "north"),
  covariance_model = "exponential",
  likelihood_method = "reml"
)
coef(fit)
vcov(fit)
logLik(fit)
fit$covariance_parameters  # partial sill, phi, nugget
fit$converged

prediction <- predict(fit, prediction_data, full_covariance = TRUE)
prediction$mean
prediction$variance
```

The formula's terms, factor levels and contrasts are retained for prediction.
The response may be absent from `prediction_data`. If training `coords` is a
numeric matrix, pass a numeric `newcoords` matrix to `predict()` as well.
Missing values, offsets and rank-deficient designs are rejected.

Set `initial = c(partial_sill, phi, nugget), training_iterations = 0` to keep
covariance parameters fixed. Otherwise they are estimated by ML or REML with
GLS coefficient profiling and `torch.optim.LBFGS` with a strong-Wolfe line search.
Optional `starts` accepts a list of triples. `parameter_bounds` accepts three
lower/upper pairs in physical units.

Predictions include coefficient estimation uncertainty. They target the latent
spatial process by default; use `observation_noise = TRUE` for independently
noisy future responses. Covariance parameter uncertainty is not included.
`kriging_weights(fit, prediction_data)` returns universal-kriging weights.
`logLik()` uses the fitted method.

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
