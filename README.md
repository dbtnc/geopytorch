# GeoPyTorch

[![CI](https://github.com/dbtnc/geopytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/dbtnc/geopytorch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/geopytorch.svg)](https://pypi.org/project/geopytorch)


## R package

The R interface is in `r-package/geopytorch-r`. See its
[README](https://github.com/dbtnc/geopytorch/blob/main/r-package/geopytorch-r/README.md)
for how to use it from R.

## Installation

```sh
pip install geopytorch
```

or, in a uv project:

```sh
uv add geopytorch
```

GeoPyTorch needs Python 3.12 or later. On Intel Macs, use Python 3.12 (with
uv, create the project with `uv init --python 3.12`): the last PyTorch release
for Intel Macs does not support newer versions.

## Quick start

```python
import torch

from geopytorch import OrdinaryKriging, SimpleKriging

train_x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
train_y = torch.tensor([1.2, 2.0, 0.7, 1.8])
target_x = torch.tensor([[0.25, 0.40], [0.75, 0.60]])

simple = SimpleKriging(mean=1.4, ard=True).fit(train_x, train_y)
simple_prediction = simple.predict(target_x)

ordinary = OrdinaryKriging(ard=True).fit(train_x, train_y)
ordinary_prediction = ordinary.predict(target_x)

print(ordinary_prediction.mean)
print(ordinary_prediction.standard_deviation)
```

Choose a covariance model with `covariance_model`:

```python
ordinary = OrdinaryKriging(covariance_model="spherical").fit(train_x, train_y)
```

Supported models are `"gaussian"`, `"exponential"`, `"matern"`,
`"spherical"`, and `"wave"`.

## Covariance kernels

The built-in kernels use the geostatistical range parameter `phi` directly:

| Model | Correlation at distance `h` |
|---|---|
| Gaussian | `exp(-(h / phi) ** 2)` |
| Exponential | `exp(-h / phi)` |
| Matérn | Matérn with Bessel argument `h / phi` |
| Spherical | `1 - 1.5 h/phi + 0.5 (h/phi) ** 3` for `h < phi`; zero otherwise |
| Wave | `(phi / h) sin(h / phi)`, with value one at `h = 0` |

For Matérn, `matern_nu` is the smoothness parameter; the supported values are
`0.5`, `1.5`, and `2.5`. Spherical and wave kernels are restricted to at most
three spatial dimensions, covering the current 2D and planned 3D use cases.

`predict()` returns a `KrigingPrediction` containing the joint covariance as
well as marginal variance and standard deviation. It predicts the latent
spatial signal by default. Use `observation_noise=True` to include the fitted
nugget in a future noisy observation:

```python
observed_prediction = ordinary.predict(target_x, observation_noise=True)
lower, upper = observed_prediction.confidence_region()
```

## Covariance parameter estimation

`OrdinaryKriging` supports both maximum likelihood and restricted maximum
likelihood. REML profiles the unknown constant mean out of a dense Gaussian
restricted-likelihood objective. L-BFGS is available for deterministic
likelihood optimization:

```python
ordinary = OrdinaryKriging(covariance_model="exponential").fit(
    train_x,
    train_y,
    likelihood_method="reml",
    optimizer="lbfgs",
    training_iterations=50,
    learning_rate=1.0,
)
print(ordinary.restricted_log_likelihood_)
```

REML is not exposed for `SimpleKriging`: its constant mean is known rather
than an unknown fixed effect. Use the default maximum-likelihood objective to
estimate its covariance parameters.

## Known covariance parameters

Both estimators default to a scaled Matérn-3/2 kernel and a
homoskedastic Gaussian likelihood. To perform classical Kriging with parameters
supplied by the user, construct a covariance kernel and skip likelihood
optimization:

```python
import gpytorch
from geopytorch import covariance_kernel

kernel = covariance_kernel("matern", matern_nu=1.5, ard_num_dims=2)
kernel.initialize(outputscale=2.0)
kernel.base_kernel.initialize(lengthscale=torch.tensor([0.4, 0.8]))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.initialize(noise=0.05)

ordinary = OrdinaryKriging(kernel=kernel, likelihood=likelihood)
ordinary.fit(train_x, train_y, training_iterations=0)
```

Simple Kriging freezes its known constant mean. Ordinary Kriging uses a
trainable constant while fitting covariance hyperparameters, then applies the
exact unit-sum weight and mean-uncertainty corrections during prediction.
