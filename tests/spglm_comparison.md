# Spatial GLM comparison

CPU float64; spmodel 0.14.0, GPyTorch 1.15.2, PyTorch 2.2.2.

Regenerate from the project root with `Rscript tests/generate_spglm_reference.R` and `PYTHONPATH=src python tests/compare_spglm.py`. The R generator requires spmodel and jsonlite. The Python tests use the saved reference fixture and do not require R.

The 60 fixed-parameter cases cover six families, five existing covariance models, and ML/REML. They include offsets, nonconstant binomial totals, coefficient covariance, predictions, and delta-method variances. Reference calls use `local = FALSE`, `diagtol = 0`, and `delta = TRUE` for response uncertainty. Matérn smoothness is 1.5; the reference range is sqrt(3) times phi.

Maximum absolute errors across fixed-parameter cases:

| Family | Coefficients | Coefficient covariance | Log likelihood | Prediction means | Prediction variances |
|---|---:|---:|---:|---:|---:|
| poisson | 5.93e-13 | 1.52e-08 | 1.9e-06 | 8.53e-12 | 4.01e-11 |
| nbinomial | 3.33e-15 | 5.02e-10 | 1.52e-08 | 7.11e-15 | 1.21e-13 |
| binomial | 2.09e-10 | 5.48e-07 | 2.05e-05 | 1.2e-10 | 1.23e-10 |
| beta | 3.67e-11 | 3.52e-07 | 1.29e-05 | 3.33e-11 | 2.34e-11 |
| Gamma | 1.39e-10 | 4.27e-07 | 2.42e-05 | 1.18e-10 | 3.84e-10 |
| inverse.gaussian | 2.11e-15 | 1.52e-10 | 9.67e-09 | 5.33e-15 | 2.13e-14 |

Twelve further cases estimate partial sill and dispersion with range and nugget fixed. Several reach zero-sill or infinite-dispersion limits. Maximum coefficient and log-likelihood errors are 3.56e-06 and 5.64e-05, respectively. At infinite dispersion, the inverse dispersion and fitted results are useful comparisons; raw dispersion estimates differ.

The beta REML boundary case retains a finite iterate but reports nonconvergence when the mode score loses numerical precision. The reference reaches its dispersion cap of 1e8. A separate exploratory fit with all covariance parameters free also drives beta dispersion above 4e7; see `zscratch/inspect_glm_fit.py`. These are small examples, not evidence of universal optimizer reliability.

Eight cases estimate only dispersion with fixed covariance and have finite interior optima:

| Family | Method | GeoPyTorch dispersion | Reference dispersion |
|---|---|---:|---:|
| nbinomial | ml | 30.2481 | 30.2481 |
| nbinomial | reml | 30.2481 | 30.2481 |
| beta | ml | 9.58533 | 9.58533 |
| beta | reml | 9.58533 | 9.58533 |
| Gamma | ml | 3.3585 | 3.3585 |
| Gamma | reml | 3.3585 | 3.3585 |
| inverse.gaussian | ml | 2.50327 | 2.50327 |
| inverse.gaussian | reml | 2.50327 | 2.50327 |

For the dispersion-only inverse-Gaussian examples, direct reference optimization failed with a convergence error. The generator instead uses R's `optimize()` over log dispersion in log([0.5, 20]), evaluating the fixed-parameter `spmodel::spglm()` likelihood. This fallback is recorded in the fixture. The other dispersion-only references use the package's own optimizer.

Prediction means use the inverse link at the predicted latent mean. Response variances use the delta method; they are not integrated posterior moments. Covariance and dispersion estimation uncertainty is excluded. The comparison retains the reference profile-Laplace normalizing constant. No grouped random effects, partition factors, anisotropy, or local approximation are tested.
